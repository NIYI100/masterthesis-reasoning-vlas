"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        save_every_n_steps: Optional[int] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # how often to save checkpoints
        self.save_every_n_steps = save_every_n_steps
        if save_every_n_steps is not None:
            assert save_every_n_steps > 0

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

        self.ACTION_TOKEN_IDS = self.vlm.llm_backbone.tokenizer.encode("ACTION: ", add_special_tokens=False)

    @property
    def _model(self):
        """Return the underlying model for attribute access. DDP wraps the model and does not forward
        attribute access (e.g. vision_backbone, llm_backbone) in many PyTorch versions; use this
        property when accessing submodules. For forward/parameters/train(), use self.vlm directly."""
        if hasattr(self.vlm, "module"):
            return self.vlm.module
        return self.vlm

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return
                        elif (
                            self.save_every_n_steps is not None
                            and (metrics.global_step + 1) % self.save_every_n_steps == 0
                        ):

                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for step, batch in enumerate(dataloader):
                # --- DATA VERIFICATION LOGGING ---
                if step % 500 == 0:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    
                    lang = batch.get("language_instruction", ["Unknown"])[0]
                    reasoning = batch.get("reasoning", ["Unknown"])[0]
                    
                    log_filename = f"gpu_{rank}_data_fingerprint.txt"
                    with open(log_filename, "a", encoding="utf-8") as f:
                        f.write(f"--- GPU Rank: {rank} | Step: {step} ---\n")
                        f.write(f"Task     : {lang}\n")
                        f.write(f"Reasoning: {reasoning}\n\n")


                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Alignments ===
                action_preds = output.logits[:, self._model.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                
                  # Replace with your actual <ACTION> token ID
                
                # === 1. Create Masks for Each Zone ===
                
                # Valid Mask: All tokens that are NOT padding (-100)
                valid_mask = (action_gt != -100)
                
                # Action Mask: Only the continuous action dimension tokens
                action_mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx)
                
                # Tag Mask: Exactly the <ACTION> tag
                #tag_mask = (action_gt == self.ACTION_TOKEN_IDS)

                seq_len = action_gt.shape[1]
                positions = torch.arange(seq_len, device=action_gt.device).unsqueeze(0).expand(action_gt.shape[0], -1)
                # 1. Initialize an empty boolean mask
                has_action = action_mask.any(dim=1, keepdim=True)
                first_action_idx = action_mask.long().argmax(dim=1, keepdim=True)
                
                # Tag Mask: Exactly the N tokens before the first action token
                tag_len = len(self.ACTION_TOKEN_IDS)
                tag_mask = (positions >= (first_action_idx - tag_len)) & (positions < first_action_idx)
                tag_mask = tag_mask & has_action & valid_mask
                #tag_mask = torch.zeros_like(action_gt, dtype=torch.bool)
                
                # 2. Shifted Tensor Match (Blazingly fast for exactly 3 tokens)
                # action_gt shape: [batch, seq_len]
                # We check: Token 1 == 41895 AND Token 2 == 25 AND Token 3 == 220
                #match_starts = (
                #    (action_gt[:, :-2] == 41895) & 
                #    (action_gt[:, 1:-1] == 25) & 
                #    (action_gt[:, 2:] == 220)
                #)
                
                # 3. Expand the match to cover all 3 token positions in the mask
                # If match_starts is True at index i, we set i, i+1, and i+2 to True
                #tag_mask[:, :-2] |= match_starts
                #tag_mask[:, 1:-1] |= match_starts
                #tag_mask[:, 2:] |= match_starts
                
                # Reasoning Mask: Valid tokens that are NOT actions and NOT the action tag
                #reasoning_mask = valid_mask & ~action_mask & ~tag_mask
                reasoning_mask = (positions < (first_action_idx - tag_len)) & has_action & valid_mask
                #reasoning_mask = torch.where(has_action, reasoning_mask, valid_mask)

                # === 2. Calculate Accuracies ===
                
                # Whole Accuracy (Everything valid)
                whole_correct = (action_preds == action_gt) & valid_mask
                whole_accuracy = whole_correct.sum().float() / torch.clamp(valid_mask.sum().float(), min=1e-5)
                
                # Reasoning Accuracy (CoT / Plan)
                reasoning_correct = (action_preds == action_gt) & reasoning_mask
                reasoning_accuracy = reasoning_correct.sum().float() / torch.clamp(reasoning_mask.sum().float(), min=1e-5)
                
                # Action Accuracy (Continuous Actions)
                action_correct = (action_preds == action_gt) & action_mask
                action_accuracy = action_correct.sum().float() / torch.clamp(action_mask.sum().float(), min=1e-5)
                
                # Action Tag Accuracy
                tag_correct = (action_preds == action_gt) & tag_mask
                action_tag_accuracy = tag_correct.sum().float() / torch.clamp(tag_mask.sum().float(), min=1e-5)

                decoded_preds = action_tokenizer.decode_token_ids_to_actions(action_preds[action_mask].cpu().numpy())
                decoded_gt = action_tokenizer.decode_token_ids_to_actions(action_gt[action_mask].cpu().numpy())

                # === Compute L1 Loss on Predicted (Continuous) Actions ===
                continuous_actions_pred = decoded_preds.clone().detach() if torch.is_tensor(decoded_preds) else torch.tensor(decoded_preds)
                continuous_actions_gt = decoded_gt.clone().detach() if torch.is_tensor(decoded_gt) else torch.tensor(decoded_gt)
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                """
                debug_print_interval = 5000
                if step % debug_print_interval == 0 and overwatch.is_rank_zero():
                    tokenizer = self.vlm.llm_backbone.tokenizer
                    gt_seq = action_gt[0]
                    pred_seq = action_preds[0]
                    
                    # 1. Extract Raw Token IDs (as lists)
                    gt_reasoning_ids = gt_seq[reasoning_mask[0]].tolist()
                    gt_tag_ids = gt_seq[tag_mask[0]].tolist()
                    gt_action_ids = gt_seq[action_mask[0]].tolist()
                    
                    pred_reasoning_ids = pred_seq[reasoning_mask[0]].tolist()
                    pred_tag_ids = pred_seq[tag_mask[0]].tolist()
                    pred_action_ids = pred_seq[action_mask[0]].tolist()
                    
                    # 2. Decode Tokens to Text
                    gt_reasoning_text = tokenizer.decode(gt_reasoning_ids, skip_special_tokens=False)
                    gt_tag_text = tokenizer.decode(gt_tag_ids, skip_special_tokens=False)
                    gt_action_text = tokenizer.decode(gt_action_ids, skip_special_tokens=False)
                    
                    pred_reasoning_text = tokenizer.decode(pred_reasoning_ids, skip_special_tokens=False)
                    pred_tag_text = tokenizer.decode(pred_tag_ids, skip_special_tokens=False)
                    pred_action_text = tokenizer.decode(pred_action_ids, skip_special_tokens=False)
                    
                    item0_reasoning_acc = reasoning_correct[0].sum().float() / torch.clamp(reasoning_mask[0].sum().float(), min=1e-5)
                    item0_tag_acc = tag_correct[0].sum().float() / torch.clamp(tag_mask[0].sum().float(), min=1e-5)
                    item0_action_acc = action_correct[0].sum().float() / torch.clamp(action_mask[0].sum().float(), min=1e-5)

                    print("\n" + "="*80)
                    print(f"=== MASK ALIGNMENT & TOKEN DEBUGGER (Step {step}, Item 0) ===")
                    print("="*80)
                    print("[REASONING MASK]")
                    print(f"  GT   IDs : {gt_reasoning_ids}")
                    print(f"  GT   Text: {gt_reasoning_text}")
                    print(f"  PRED IDs : {pred_reasoning_ids}")
                    print(f"  PRED Text: {pred_reasoning_text}")
                    print(f"  --> Item 0 Reasoning Acc: {item0_reasoning_acc.item():.4f}")
                    print("-" * 80)
                    print("[TAG MASK]")
                    print(f"  GT   IDs : {gt_tag_ids}")
                    print(f"  GT   Text: '{gt_tag_text}'")
                    print(f"  PRED IDs : {pred_tag_ids}")
                    print(f"  PRED Text: '{pred_tag_text}'")
                    print(f"  --> Item 0 Tag Acc: {item0_tag_acc.item():.4f}")
                    print("-" * 80)
                    print("[ACTION MASK]")
                    print(f"  GT   IDs : {gt_action_ids}")
                    print(f"  GT   Text: {gt_action_text}")
                    print(f"  PRED IDs : {pred_action_ids}")
                    print(f"  PRED Text: {pred_action_text}")
                    print(f"  --> Item 0 Action Acc: {item0_action_acc.item():.4f}")
                    print("="*80 + "\n")
                # -----------------------------------------
                """
                # === Commit Metrics ===
                metrics.commit(
                    whole_accuracy=whole_accuracy,
                    reasoning_accuracy=reasoning_accuracy,
                    action_accuracy=action_accuracy, 
                    action_tag_accuracy=action_tag_accuracy,
                    l1_loss=action_l1_loss, 
                    update_step_time=True
                )

                """
                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx)

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                """





                """
                save_inference_interval = 10000  # Passe diese Zahl nach Bedarf an

                if metrics.global_step % save_inference_interval == 0:
                    
                    # --- 1. CRITICAL: All ranks must execute this block ---
                    # Switch to eval mode to disable dropout, etc.
                    self.vlm.eval()
                    
                    # Extract just the prompt tokens to feed into generate.
                    # We find where the labels start (are not -100) and slice the input_ids.
                    first_target_idx = (batch["labels"][0] != -100).nonzero(as_tuple=True)[0][0]
                    prompt_input_ids = batch["input_ids"][:, :first_target_idx]
                    
                    with torch.inference_mode(): # Prevents OOM by disabling gradients
                        generated_output = self.vlm.llm_backbone.generate(
                            input_ids=prompt_input_ids,
                            max_new_tokens=64, # Adjust based on how long your expected actions/text are
                            use_cache=True
                        )
                    
                    # Switch back to train mode immediately after
                    self.vlm.train()
                    
                    # --- 2. CRITICAL: Only Rank 0 saves the file ---
                    if overwatch.is_rank_zero():
                        import json
                        from pathlib import Path
                        
                        tokenizer = self.vlm.llm_backbone.tokenizer
                        
                        # Decode the realistic generated text
                        # generated_output contains both prompt and new tokens. We only want the new ones.
                        new_tokens = generated_output[0][first_target_idx:]
                        realistic_predicted_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        
                        # Decode Ground Truth (as you did before)
                        full_gt_ids = batch["labels"][0]
                        valid_gt_ids = full_gt_ids[full_gt_ids != -100]
                        full_gt_text = tokenizer.decode(valid_gt_ids, skip_special_tokens=True)
                        
                        # Pack data
                        inference_data = {
                            "training_state": {
                                "global_step": metrics.global_step,
                                "total_loss": loss.item(),
                                "action_l1_loss": action_l1_loss.item(),
                            },
                            "text_data": {
                                "realistic_predicted_text": realistic_predicted_text,
                                "ground_truth_text": full_gt_text
                            }
                            # Note: You may need custom parsing here to extract continuous 
                            # actions from 'realistic_predicted_text' if your model outputs 
                            # actions as special tokens within the text stream.
                        }
                        
                        # Create folder and save
                        inference_dir = Path(metrics.run_dir) / "inferences"
                        inference_dir.mkdir(parents=True, exist_ok=True)
                        
                        save_path = inference_dir / f"inference_step_{metrics.global_step}.json"
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(inference_data, f, indent=4)
                


                save_inference_interval = 10000  # Passe diese Zahl nach Bedarf an
                
                if overwatch.is_rank_zero() and (metrics.global_step % save_inference_interval == 0):
                    import json
                    
                    tokenizer = self.vlm.llm_backbone.tokenizer
                    full_pred_ids = output.logits[0].argmax(dim=-1)
                    full_predicted_text = tokenizer.decode(full_pred_ids, skip_special_tokens=True)
                    
                    # 3. Gesamte Ground Truth (Text) für das ERSTE Element dekodieren
                    # WICHTIG: Die Labels enthalten -100 für Tokens, die nicht trainiert werden (z.B. der Prompt).
                    # Diese müssen wir vor dem Dekodieren herausfiltern.
                    full_gt_ids = batch["labels"][0]
                    valid_gt_ids = full_gt_ids[full_gt_ids != -100]
                    full_gt_text = tokenizer.decode(valid_gt_ids, skip_special_tokens=True)
                    
                    single_action_pred = continuous_actions_pred[0].tolist() if len(continuous_actions_pred) > 0 else []
                    single_action_gt = continuous_actions_gt[0].tolist() if len(continuous_actions_gt) > 0 else []

                    
                    # Alle gewünschten Daten und States in ein Dictionary packen
                    inference_data = {
                        "training_state": {
                            "global_step": metrics.global_step,
                            "total_loss": loss.item(),
                            "action_l1_loss": action_l1_loss.item(),
                        },
                        "text_data": {
                            "predicted_text": full_predicted_text,
                            "ground_truth_text": full_gt_text
                        },
                        "action_vectors": {
                            "predicted_action": single_action_pred,
                            "ground_truth_action": single_action_gt
                        }
                    }
                    
                    # Ordner erstellen und speichern
                    inference_dir = Path(metrics.run_dir) / "inferences"
                    inference_dir.mkdir(parents=True, exist_ok=True)
                    
                    save_path = inference_dir / f"inference_step_{metrics.global_step}.json"
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(inference_data, f, indent=4)

                """



                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
