import json
import re
import ast
import numpy as np
import os

"""
Reasoning looks like this:

PLAN:@{'0': 'move to the black bowl', '1': 'grasp the black bowl', '2': 'move the black bowl to the plate', '3': 'release the black bowl onto the plate'}
@VISIBLE OBJECTS:@{'akita black bowl 1': [[147, 70], [176, 104]], 'akita black bowl 2': [[105, 148], [119, 178]], 'new salad dressing 1': [[71, 70], [109, 87]], 'chocolate pudding 1': [[144, 126], [169, 141]], 'wooden cabinet 1': [[77, 0], [176, 59]]}
@SUBTASK REASONING:@The robot needs to move to the black bowl because it is currently positioned far away from it, while the bowl is located to the right of the plate.
@SUBTASK:@move to the black bowl
@MOVE REASONING:@The black bowl is positioned on the left side of the table, while the plate is on the right, so the robot should move back to create space and reach the bowl.
@MOVE:@move back slowly
@GRIPPER POSITION:@[37, 110] 
ACTION: <|extra_178|><|extra_142|><|extra_106|><|extra_9|><|extra_134|><|extra_254|><|extra_25|><|im_end|><|endoftext|>
"""

def get_reasoning_fn(fn_name: str):
    """
    Looks up a function defined in this file by its string name.
    """
    if not fn_name or fn_name.lower() in ["none", "null", ""]:
        return None
        
    # Get the dictionary of everything defined in THIS file
    current_file_namespace = globals()
    
    if fn_name in current_file_namespace:
        return current_file_namespace[fn_name]
    else:
        print(f"Warning: Function '{fn_name}' not found in reasoning_manipulation.py")
        return None

def no_reasoning(reasoning):
    return ""


def reasoning_dropout_50(reasoning):
    """
    ECoT-Lite: Randomly drop reasoning with 50% probability during training.
    At inference, use reasoning_modifier_fn_str="None" so no reasoning is generated.
    This matches the paper's best LIBERO-90 results (~89%).
    """
    import random
    return "" if random.random() < 0.5 else reasoning

def swap_x_y(reasoning):
    # Replacement function for the VISIBLE OBJECTS dictionary
    def swap_x_y_objects(match):
        dict_str = match.group(1)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            # Swap [x, y] to [y, x]
            objects_dict[key] = [[c[1], c[0]] for c in coords]
            
        # Reconstruct the matched portion
        return f"VISIBLE OBJECTS:@{str(objects_dict)}@"

    # Replacement function for the GRIPPER POSITION list
    def swap_x_y_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        
        # Swap [x, y] to [y, x]
        if len(gripper_list) == 2:
            gripper_list = [gripper_list[1], gripper_list[0]]
            
        # Reconstruct the matched portion
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    # Apply the replacements
    # Match everything between 'VISIBLE OBJECTS:@' and the next '@'
    updated_string = re.sub(r'VISIBLE OBJECTS:@(\{.*?\})@', swap_x_y_objects, reasoning)
    
    # Match the coordinate list after 'GRIPPER POSITION:@'
    updated_string = re.sub(r'GRIPPER POSITION:@(\[.*?\])', swap_x_y_gripper, updated_string)

    return updated_string

def _move_bboxes(reasoning, offset):
    def adjust_objects(match):
        dict_str = match.group(1)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            # Add the offset to x and y for both points [[x1, y1], [x2, y2]]
            objects_dict[key] = [[c[0] + offset, c[1] + offset] for c in coords]
            
        # Reconstruct the matched portion
        return f"VISIBLE OBJECTS:@{str(objects_dict)}@"

    def adjust_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        
        # Add the offset to x and y
        if len(gripper_list) == 2:
            gripper_list = [gripper_list[0] + offset, gripper_list[1] + offset]
            
        # Reconstruct the matched portion
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    updated_string = re.sub(r'VISIBLE OBJECTS:@(\{.*?\})@', adjust_objects, reasoning)
    updated_string = re.sub(r'GRIPPER POSITION:@(\[.*?\])', adjust_gripper, updated_string)

    return updated_string

def move_bboxes_15(reasoning):
    return _move_bboxes(reasoning, 15)

def _cut_out_of_reasoning(reasoning, reasoning_tag):
    """
    Removes a specific tag and its contents from the reasoning string.
    """
    pattern = r"@?" + re.escape(reasoning_tag) + r":@?.*?(?=@|\sACTION:|$)"
    
    cleaned_text = re.sub(pattern, "", reasoning)
    return cleaned_text.lstrip('@')

def cut_out_gripper(reasoning):
    return _cut_out_of_reasoning(reasoning, "GRIPPER POSITION")

def cut_out_move(reasoning):
    return _cut_out_of_reasoning(reasoning, "MOVE")

def cut_out_move_reasoning(reasoning):
    return _cut_out_of_reasoning(reasoning, "MOVE REASONING")

def cut_out_subtask(reasoning):
    return _cut_out_of_reasoning(reasoning, "SUBTASK")

def cut_out_subtask_reasoning(reasoning):
    return _cut_out_of_reasoning(reasoning, "SUBTASK REASONING")

def cut_out_visible_objects(reasoning):
    return _cut_out_of_reasoning(reasoning, "VISIBLE OBJECTS")

def cut_out_plan(reasoning):
    return _cut_out_of_reasoning(reasoning, "PLAN")

def gauß_50(reasoning):
    return _gauß_on_bboxes(reasoning, 50, "/home/hk-project-p0024638/uvrfq/shifts_sigma50.jsonl")

def _gauß_on_bboxes(reasoning, sigma, folder_path, max_val=224):
    actual_dx = []
    actual_dy = []
    # Helper function to apply Gaussian shift and track it
    def apply_gauss_to_point(x, y):
        intended_dx = int(np.round(np.random.normal(0, sigma)))
        intended_dy = int(np.round(np.random.normal(0, sigma)))
        
        new_x = max(0, min(x + intended_dx, max_val))
        new_y = max(0, min(y + intended_dy, max_val))

        actual_dx.append(new_x - x)
        actual_dy.append(new_y - y)
        
        return new_x, new_y

    # Replacement function for the VISIBLE OBJECTS dictionary
    def shift_objects(match):
        dict_str = match.group(1)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            new_coords = []
            for c in coords:
                # Calculate the Gaussian shift for every individual point
                new_x, new_y = apply_gauss_to_point(c[0], c[1])
                new_coords.append([new_x, new_y])
            objects_dict[key] = new_coords
            
        return f"VISIBLE OBJECTS:@{str(objects_dict)}@"

    # Replacement function for the GRIPPER POSITION list
    def shift_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        
        if len(gripper_list) == 2:
            new_x, new_y = apply_gauss_to_point(gripper_list[0], gripper_list[1])
            gripper_list = [new_x, new_y]
            
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    updated_string = re.sub(r'VISIBLE OBJECTS:@(\{.*?\})@', shift_objects, reasoning)
    updated_string = re.sub(r'GRIPPER POSITION:@(\[.*?\])', shift_gripper, updated_string)

    if actual_dx and actual_dy:
        update_running_sigma(folder_path, sigma, actual_dx, actual_dy)

    return updated_string

def initialize_shift_log(filepath, target_sigma):
    """
    Creates the directory and initializes a .jsonl file with the target sigma.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Open in 'w' mode to create a fresh file (overwrites existing)
    with open(filepath, 'w') as f:
        metadata = {"target_sigma": target_sigma, "type": "metadata"}
        f.write(json.dumps(metadata) + '\n')
        
    print(f"Initialized log file at: {filepath}")

def calculate_real_sigma_from_log(filepath):
    """
    Reads the .jsonl file line by line and calculates the true mean and std dev.
    """
    all_dx = []
    all_dy = []
    target_sigma = None

    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f):
            data = json.loads(line.strip())
            
            # The first line contains our metadata
            if line_number == 0 and data.get("type") == "metadata":
                target_sigma = data.get("target_sigma")
                continue
            
            # Extend our flattened lists with the values from this step
            all_dx.extend(data.get("dx", []))
            all_dy.extend(data.get("dy", []))

    # Convert to numpy arrays for fast math
    dx_array = np.array(all_dx)
    dy_array = np.array(all_dy)

    # Calculate statistics
    real_mean_x = np.mean(dx_array) if len(dx_array) > 0 else 0
    real_sigma_x = np.std(dx_array, ddof=1) if len(dx_array) > 1 else 0

    real_mean_y = np.mean(dy_array) if len(dy_array) > 0 else 0
    real_sigma_y = np.std(dy_array, ddof=1) if len(dy_array) > 1 else 0

    print(f"--- Results for {filepath} ---")
    print(f"Target Sigma: {target_sigma}")
    print(f"Total points shifted: {len(dx_array)}")
    print(f"Real X -> Mean: {real_mean_x:.2f}, Sigma: {real_sigma_x:.2f}")
    print(f"Real Y -> Mean: {real_mean_y:.2f}, Sigma: {real_sigma_y:.2f}")
    
    return real_sigma_x, real_sigma_y

def update_running_sigma(folder_path, target_sigma, actual_dx, actual_dy):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, f"running_sigma_{target_sigma}.json")
    
    # 1. Load existing state or initialize a new one
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
    else:
        state = {
            "target_sigma": target_sigma,
            "n": 0,
            "sum_dx": 0.0,
            "sum_dy": 0.0,
            "sum_sq_dx": 0.0,
            "sum_sq_dy": 0.0,
            "real_sigma_x": 0.0,
            "real_sigma_y": 0.0
        }
        
    # 2. If no shifts occurred in this call, just return current sigmas
    if not actual_dx or not actual_dy:
        return state["real_sigma_x"], state["real_sigma_y"]
        
    # 3. Update running totals
    n_new = len(actual_dx)
    dx_arr = np.array(actual_dx)
    dy_arr = np.array(actual_dy)
    
    state["n"] += n_new
    state["sum_dx"] += float(np.sum(dx_arr))
    state["sum_dy"] += float(np.sum(dy_arr))
    state["sum_sq_dx"] += float(np.sum(dx_arr**2))
    state["sum_sq_dy"] += float(np.sum(dy_arr**2))
    
    # 4. Calculate the new running standard deviation
    n = state["n"]
    if n > 1:
        mean_x = state["sum_dx"] / n
        mean_y = state["sum_dy"] / n
        
        # Population variance
        var_x = (state["sum_sq_dx"] / n) - (mean_x ** 2)
        var_y = (state["sum_sq_dy"] / n) - (mean_y ** 2)
        
        # Apply Bessel's correction (n / (n-1)) for Sample variance
        var_x = var_x * (n / (n - 1))
        var_y = var_y * (n / (n - 1))
        
        # max(0, var) prevents crashing from microscopic negative floats due to precision limits
        state["real_sigma_x"] = float(np.sqrt(max(0, var_x)))
        state["real_sigma_y"] = float(np.sqrt(max(0, var_y)))
        
    # 5. Save the updated state safely
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=4)
        
    return state["real_sigma_x"], state["real_sigma_y"]

# TODO - Eigentlich sollte jetzt
if __name__ == "__main__":
    log_filepath = "./shifts_sigma50.jsonl"
    target_sigma = 50
    initialize_shift_log(log_filepath, target_sigma)
    reasoning_output = "VISIBLE OBJECTS:@{'apple': [[0, 5], [150, 150]]}@ GRIPPER POSITION:@[100, 0]"
    altered_reasoning = _gauß_on_bboxes(reasoning_output, target_sigma, log_filepath)
    print(reasoning_output)
    print(altered_reasoning)
    print(calculate_real_sigma_from_log(log_filepath))