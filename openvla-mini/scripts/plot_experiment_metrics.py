#!/usr/bin/env python3
"""
Dynamic plotting for experiment JSON logs.

This script reads eval JSON metrics (typically *GLOBAL-RESULTS.json),
flattens nested keys, applies plot specs from YAML, and writes static
charts (PNG/PDF/SVG) via Matplotlib for headless / server use.

Dependencies: matplotlib, pyyaml (see requirements-plot.txt).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: matplotlib. Install with `pip install matplotlib` "
        "or `pip install -r scripts/requirements-plot.txt`."
    ) from exc

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml. Install with `pip install pyyaml`."
    ) from exc


DEFAULT_PATTERN = "*GLOBAL-RESULTS.json"

# CoT sections that appear under reasoning_metrics.text_rouge_l.* in GLOBAL-RESULTS.json
TEXT_WORD_DROPOUT_FIELDS: tuple[str, ...] = (
    "plan",
    "subtask_reasoning",
    "subtask",
    "move_reasoning",
    "move",
    "whole",
)


@dataclass
class PlotSpec:
    plot_id: str
    title: str
    chart_type: str
    x: str
    y: str
    color: str | None = None
    facet: str | None = None
    sort_by_x: bool = False
    filters: dict[str, Any] | None = None
    x_label: str | None = None
    y_label: str | None = None
    color_legend_title: str | None = None
    category: str | None = None  # bbox | gripper | text — for --noise-* CLI filters


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        nk = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, nk))
        else:
            out[nk] = value
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_sigma_from_text(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"sigma[_=:-]?([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        return _to_float(m.group(1))
    p = re.search(r"\bp([0-9]{1,3})\b", text)
    if p:
        return _to_float(p.group(1)) / 100.0
    return None


def derive_line_series(rec: dict[str, Any]) -> str:
    """
    Group runs that should be connected as one polyline: same bbox/gripper sweep per suite,
    or same word-dropout target (plan / move / …) across p10…p40.
    """
    et = str(rec.get("experiment_type", "")).strip().lower()
    pt = str(rec.get("perturbation_type", "")).strip().lower()
    pl = str(rec.get("perturbation_level", "")).strip()
    ts = str(rec.get("task_suite_name", "default"))

    if et in ("noise_bbox", "noise_gripper"):
        return f"{ts}|{pt or et}"

    if et == "noise_text" and pt == "word_dropout":
        m = re.match(r"^p([0-9]+)_(.+)$", pl, flags=re.IGNORECASE)
        if m:
            return m.group(2).lower()

    return pl or "default"


def normalize_record(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    rec = flatten_dict(payload)
    rec["_source_path"] = str(source_path.resolve())
    rec["_source_name"] = source_path.name
    rec["_source_parent"] = source_path.parent.name

    if "noise_sigma" not in rec or rec.get("noise_sigma") is None:
        parsed = parse_sigma_from_text(str(rec.get("perturbation_level", "")))
        if parsed is not None:
            rec["noise_sigma"] = parsed

    rec["noise_sigma"] = _to_float(rec.get("noise_sigma"))
    rec["line_series"] = derive_line_series(rec)
    return rec


def collect_json_files(logs_root: Path, pattern: str) -> list[Path]:
    if not logs_root.exists():
        return []
    return sorted(p for p in logs_root.rglob(pattern) if p.is_file())


def load_records(files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        records.append(normalize_record(payload, p))
    return records


def load_plot_specs(spec_path: Path) -> list[PlotSpec]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "plots" not in raw or not isinstance(raw["plots"], list):
        raise SystemExit(f"Invalid spec file format: {spec_path}")

    specs: list[PlotSpec] = []
    for item in raw["plots"]:
        if not isinstance(item, dict):
            continue
        specs.append(
            PlotSpec(
                plot_id=str(item["id"]),
                title=str(item["title"]),
                chart_type=str(item.get("chart_type", "line")),
                x=str(item["x"]),
                y=str(item["y"]),
                color=item.get("color"),
                facet=item.get("facet"),
                sort_by_x=bool(item.get("sort_by_x", False)),
                filters=item.get("filters"),
                x_label=item.get("x_label"),
                y_label=item.get("y_label"),
                color_legend_title=item.get("color_legend_title"),
                category=item.get("category"),
            )
        )
    return specs


def _field_label(field: str) -> str:
    return field.replace("_", " ").title()


def build_text_word_dropout_specs(fields: tuple[str, ...] = TEXT_WORD_DROPOUT_FIELDS) -> list[PlotSpec]:
    """Per reasoning field: word-dropout probability vs SR, and ROUGE-L vs SR (word-dropout runs only)."""
    out: list[PlotSpec] = []
    for field in fields:
        rouge_x = f"reasoning_metrics.text_rouge_l.{field}.mean"
        fl = _field_label(field)
        base_filters: dict[str, Any] = {
            "experiment_type": "noise_text",
            "perturbation_type": "word_dropout",
            "line_series": field,
        }
        out.append(
            PlotSpec(
                plot_id=f"text_wd_{field}_dropout_vs_sr",
                title=f"Text word dropout ({fl}): success vs dropout probability",
                chart_type="line",
                x="noise_sigma",
                y="total_success_rate",
                color="task_suite_name",
                sort_by_x=True,
                filters=base_filters,
                x_label="Dropout probability",
                y_label="Success rate",
                color_legend_title="Task suite",
                category="text",
            )
        )
        out.append(
            PlotSpec(
                plot_id=f"text_wd_{field}_rouge_vs_sr",
                title=f"Text word dropout ({fl}): success vs ROUGE-L",
                chart_type="line",
                x=rouge_x,
                y="total_success_rate",
                color="task_suite_name",
                sort_by_x=True,
                filters=base_filters,
                x_label=f"ROUGE-L ({fl})",
                y_label="Success rate",
                color_legend_title="Task suite",
                category="text",
            )
        )
    return out


def resolve_active_categories(args: argparse.Namespace) -> set[str] | None:
    """If any --noise-* flag is set, only those categories; else None = all."""
    cats: list[str] = []
    if args.noise_bbox:
        cats.append("bbox")
    if args.noise_gripper:
        cats.append("gripper")
    if args.noise_text:
        cats.append("text")
    return set(cats) if cats else None


def _match_single_filter(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return actual in expected
    if isinstance(expected, str) and expected.startswith("re:"):
        return bool(re.search(expected[3:], str(actual)))
    return actual == expected


def apply_filters(records: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filters:
        return records
    out: list[dict[str, Any]] = []
    for rec in records:
        keep = True
        for key, expected in filters.items():
            actual = rec.get(key)
            if not _match_single_filter(actual, expected):
                keep = False
                break
        if keep:
            out.append(rec)
    return out


def is_metric_relevant(experiment_type: str | None, metric_name: str) -> bool:
    if metric_name == "total_success_rate":
        return True
    et = (experiment_type or "").strip().lower()
    if not et:
        return True
    if et == "noise_bbox":
        return metric_name.startswith("reasoning_metrics.bbox_iou")
    if et == "noise_text":
        return metric_name.startswith("reasoning_metrics.text_rouge_l")
    if et == "noise_gripper":
        return metric_name.startswith("reasoning_metrics.gripper_distance")
    return True


def any_metric_pattern_match(patterns: list[re.Pattern[str]], metric_names: list[str]) -> bool:
    if not patterns:
        return False
    return any(p.search(name) for p in patterns for name in metric_names)


def sort_records_for_x(records: list[dict[str, Any]], x_key: str) -> list[dict[str, Any]]:
    def key_fn(rec: dict[str, Any]) -> tuple[int, Any]:
        v = rec.get(x_key)
        vf = _to_float(v)
        if vf is not None:
            return (0, vf)
        return (1, "" if v is None else str(v))

    return sorted(records, key=key_fn)


def _group_records(records: list[dict[str, Any]], color_key: str | None) -> dict[str, list[dict[str, Any]]]:
    if not color_key:
        return {"all": records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get(color_key, "unknown"))].append(rec)
    return grouped


def _filter_valid_xy(records: list[dict[str, Any]], x_key: str, y_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec.get(x_key) is None or rec.get(y_key) is None:
            continue
        out.append(rec)
    return out


def _order_rows_for_plot(group_rows: list[dict[str, Any]], spec: PlotSpec) -> list[dict[str, Any]]:
    """Line charts: order by noise_sigma when it varies so sweeps follow noise level (not x-sorted)."""
    if spec.chart_type != "scatter":
        sigmas = [_to_float(r.get("noise_sigma")) for r in group_rows]
        present = [s for s in sigmas if s is not None]
        if len(present) == len(group_rows) and len(set(present)) > 1:
            return sorted(group_rows, key=lambda r: _to_float(r.get("noise_sigma")) or 0.0)
    if spec.sort_by_x:
        return sort_records_for_x(group_rows, spec.x)
    return group_rows


def build_figure(records: list[dict[str, Any]], spec: PlotSpec) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _group_records(records, spec.color)
    legend_title = spec.color_legend_title or spec.color

    for group_name, group_rows in grouped.items():
        rows = _order_rows_for_plot(group_rows, spec)
        x_vals = [r.get(spec.x) for r in rows]
        y_vals = [r.get(spec.y) for r in rows]
        label = group_name if spec.color else spec.y

        if spec.chart_type == "scatter":
            ax.scatter(x_vals, y_vals, label=label, s=36)
        else:
            ax.plot(x_vals, y_vals, marker="o", linestyle="-", label=label)

    ax.set_title(spec.title)
    ax.set_xlabel(spec.x_label or spec.x)
    ax.set_ylabel(spec.y_label or spec.y)
    ax.grid(True, alpha=0.3)
    if spec.color:
        ax.legend(title=legend_title)
    elif len(grouped) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    default_spec = Path(__file__).resolve().parent / "plot_specs.yaml"
    parser = argparse.ArgumentParser(
        description="Plot experiment metrics from JSON logs.",
        epilog="If none of --noise-bbox / --noise-gripper / --noise-text are set, every "
        "category that has data is plotted. Combine flags to select multiple categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--logs-root", type=Path, required=True, help="Root directory containing JSON log files.")
    parser.add_argument("--spec-file", type=Path, default=default_spec, help="YAML plot specification file.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Recursive glob pattern under logs-root.")
    parser.add_argument(
        "--include-experiment-type",
        nargs="*",
        default=[],
        help="Optional allowlist (e.g. noise_bbox noise_text).",
    )
    parser.add_argument(
        "--exclude-metric-pattern",
        nargs="*",
        default=[],
        help="Regex patterns. Skip specs when x or y matches one of these patterns.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for chart image files.")
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output image format (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=175, help="Raster resolution for PNG (ignored for pdf/svg).")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write manifest.json listing id, title, and file for each plot.",
    )
    parser.add_argument(
        "--noise-bbox",
        dest="noise_bbox",
        action="store_true",
        help="Only bounding-box noise plots (σ vs SR, IoU vs SR).",
    )
    parser.add_argument(
        "--noise-gripper",
        dest="noise_gripper",
        action="store_true",
        help="Only gripper noise plots (σ vs SR, mean gripper distance vs SR).",
    )
    parser.add_argument(
        "--noise-text",
        dest="noise_text",
        action="store_true",
        help="Only text word-dropout plots (per CoT field: dropout vs SR, ROUGE-L vs SR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = collect_json_files(args.logs_root, args.pattern)
    if not files:
        raise SystemExit(
            f"No files matched pattern {args.pattern!r} under {args.logs_root.resolve()}"
        )

    records = load_records(files)
    if args.include_experiment_type:
        allowed = set(args.include_experiment_type)
        records = [r for r in records if str(r.get("experiment_type")) in allowed]
    if not records:
        raise SystemExit("No records available after loading/filtering.")

    specs = load_plot_specs(args.spec_file) + build_text_word_dropout_specs()
    active_categories = resolve_active_categories(args)
    exclude_patterns = [re.compile(p) for p in args.exclude_metric_pattern]

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.logs_root / f"plot_outputs_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, str]] = []
    skipped = 0
    for spec in specs:
        if any_metric_pattern_match(exclude_patterns, [spec.x, spec.y]):
            skipped += 1
            continue

        if active_categories is not None:
            if spec.category is None or spec.category not in active_categories:
                skipped += 1
                continue

        sliced = apply_filters(records, spec.filters)
        sliced = _filter_valid_xy(sliced, spec.x, spec.y)
        if not sliced:
            skipped += 1
            continue

        if any(not is_metric_relevant(str(r.get("experiment_type")), spec.y) for r in sliced):
            sliced = [r for r in sliced if is_metric_relevant(str(r.get("experiment_type")), spec.y)]
        if not sliced:
            skipped += 1
            continue

        fig = build_figure(sliced, spec)
        out_file = out_dir / f"{spec.plot_id}.{args.format}"
        save_kw: dict[str, Any] = {"bbox_inches": "tight"}
        if args.format == "png":
            save_kw["dpi"] = args.dpi
        fig.savefig(str(out_file), **save_kw)
        plt.close(fig)
        generated.append({"id": spec.plot_id, "title": spec.title, "file": out_file.name})

    if not generated:
        raise SystemExit("No plots generated. Check filters/spec fields against available JSON keys.")

    if args.write_manifest:
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest_path.resolve()}")

    print(f"Loaded records: {len(records)}")
    print(f"Generated plots: {len(generated)}")
    print(f"Skipped specs: {skipped}")
    print(f"Output directory: {out_dir.resolve()}")
    print("Output files:")
    for i, entry in enumerate(generated, 1):
        print(f"  {i}. {(out_dir / entry['file']).resolve()}")


if __name__ == "__main__":
    main()
