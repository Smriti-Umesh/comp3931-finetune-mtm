"""
Section 4: Generalisation — does the model generalise from train to val/test?

This script loads training histories plus split-level evaluation summaries and
produces:
  - curves_train.png / curves_val.png          : overlaid training curves
  - curves_both_<label>.png                    : train + val for each run
  - generalisation_gap.csv                     : tidy split summary table
  - generalisation_bps_<mode>.png              : train/val/test BPS line plot
  - generalisation_nll_<mode>.png              : train/val/test NLL line plot

It supports both:
  - single-task runs with artifacts/split_eval_summary.json
  - combined runs with artifacts/combined_eval_summary.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


# The name of runs are set manually, as several experiments were run
# This should be changed based on new naming conventions and new experiments
# All experiment names end with JOB IDs, these have been removed for now
# and names should be updated based on new runs and naming conventions
VERIFIED_COMBINED_PRESET = [
    (
        "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5",
        "MtM 8BS 2e-5/5e-5",
    ),
    (
        "ndt1_direct_combined_full_lr2e-5",
        "NDT1 scratch 8BS",
    ),
    (
        "ndt1_stitched_combined_full_prompt_lr2e-5",
        "Stitched NDT1 8BS",
    ),
    (
        "ibl_mtm_combined_direct_full_lr1e-5_adapter5e-5_rerun",
        "MtM 1e-5 wd0.1 wu0.20",
    ),
    (
        "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd001",
        "MtM 2e-5 wd0.01 wu0.15",
    ),
    (
        "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun",
        "MtM 2e-5 wd0.01 wu0.20",
    ),
    (
        "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd01",
        "MtM 2e-5 wd0.1 wu0.15",
    ),
    (
        "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun",
        "MtM 2e-5 wd0.1 wu0.20",
    ),
    (
        "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd001",
        "MtM 3e-5 wd0.01 wu0.15",
    ),
    (
        "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun",
        "MtM 3e-5 wd0.01 wu0.20",
    ),
    (
        "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd01",
        "MtM 3e-5 wd0.1 wu0.15",
    ),
    (
        "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun",
        "MtM 3e-5 wd0.1 wu0.20",
    ),
]

PRESETS = {
    "verified_combined": VERIFIED_COMBINED_PRESET,
}




def _find(run_dir: Path, name: str):
    for candidate in [run_dir / "artifacts" / name, run_dir / name]:
        if candidate.exists():
            return candidate
    return None


def load_history(run_dir: Path):
    path = _find(run_dir, "history.csv")
    if path is None:
        raise FileNotFoundError(f"history.csv not found under {run_dir}")
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def load_metadata(run_dir: Path):
    path = _find(run_dir, "run_metadata.json")
    if path is None:
        return {}
    return json.loads(path.read_text())


def load_split_summaries(run_dir: Path):
    """
    Returns a dict mapping evaluation mode -> split summary.
    Single-task runs map to {"default": ...}; combined runs map to
    {"neuron": ..., "causal": ...}.
    """
    split_path = _find(run_dir, "split_eval_summary.json")
    if split_path is not None:
        return {"default": json.loads(split_path.read_text())}

    combined_path = _find(run_dir, "combined_eval_summary.json")
    if combined_path is None:
        return None

    combined = json.loads(combined_path.read_text())
    summaries = {}
    for mode in ("neuron", "causal"):
        if mode in combined:
            summaries[mode] = combined[mode]
    return summaries or None


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def extract_hparams(metadata):
    args = metadata.get("args", {})
    config = metadata.get("config", {})
    optimizer = config.get("optimizer", {})
    return {
        "lr": _coalesce(args.get("lr"), optimizer.get("lr")),
        "adapter_lr": args.get("adapter_lr"),
        "weight_decay": _coalesce(args.get("weight_decay"), optimizer.get("wd")),
        "warmup_pct": _coalesce(args.get("warmup_pct"), optimizer.get("warmup_pct")),
        "batch_size": args.get("batch_size"),
        "epochs": args.get("epochs"),
    }


def resolve_runs_and_labels(args):
    if args.runs and args.preset:
        raise SystemExit("Use either --runs/--labels or --preset, not both.")

    if args.preset:
        preset = PRESETS.get(args.preset)
        if preset is None:
            available = ", ".join(sorted(PRESETS))
            raise SystemExit(f"Unknown --preset '{args.preset}'. Available: {available}")
        runs = [args.results_root / rel_path for rel_path, _ in preset]
        labels = [label for _, label in preset]
        return runs, labels

    if not args.runs:
        raise SystemExit("Provide either --runs or --preset.")

    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must have the same count as --runs")
    return args.runs, labels


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _attach_legend(fig, ax, ncol=4):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(ncol, len(labels)),
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.22)


def plot_curves_overlay(runs_data, output_dir, split_key, ylabel, title, fname, plt):
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    for rd in runs_data:
        epochs = [row["epoch"] for row in rd["history"]]
        losses = [row[split_key] for row in rd["history"]]
        ax.plot(epochs, losses, linewidth=2.0, label=rd["label"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style(ax)
    _attach_legend(fig, ax, ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_single_run_both(rd, output_dir, plt):
    history = rd["history"]
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss_per_masked_bin"] for row in history]
    val = [row["val_loss_per_masked_bin"] for row in history]
    best_val = float(np.nanmin(val))
    best_epoch = epochs[int(np.nanargmin(val))]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(epochs, train, linewidth=2.0, label="train")
    ax.plot(epochs, val, linewidth=2.0, label="val")
    ax.scatter(
        [best_epoch],
        [best_val],
        s=60,
        zorder=5,
        label=f"best val={best_val:.4f} (ep {best_epoch})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per masked bin")
    ax.set_title(f"{rd['label']}: training history")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    slug = rd["label"].replace(" ", "_").replace("/", "-")
    fig.savefig(output_dir / f"curves_both_{slug}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_lines(gap_rows, output_dir, mode, metric, ylabel, title, fname, plt):
    rows = [row for row in gap_rows if row["mode"] == mode]
    if not rows:
        return

    split_names = ["train", "val", "test"]
    x = np.arange(len(split_names))
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    for row in rows:
        values = [row[f"{split}_{metric}"] for split in split_names]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=row["label"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([name.title() for name in split_names])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    _style(ax)
    _attach_legend(fig, ax, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Generalization summary
def collect_gap_data(runs_data):
    rows = []
    for rd in runs_data:
        summaries = rd.get("split_summaries")
        if summaries is None:
            continue
        for mode, summary in summaries.items():
            row = {
                "label": rd["label"],
                "run_name": rd["run_name"],
                "mode": mode,
                **rd["hparams"],
            }
            for split in ("train", "val", "test"):
                split_metrics = summary.get(split, {})
                row[f"{split}_bps"] = split_metrics.get(
                    "bits_per_spike_vs_per_neuron_mean", float("nan")
                )
                row[f"{split}_nll"] = split_metrics.get(
                    "model_nll_per_masked_bin", float("nan")
                )
            row["gap_bps"] = row["test_bps"] - row["train_bps"]
            row["gap_nll"] = row["test_nll"] - row["train_nll"]
            rows.append(row)
    return rows


def write_gap_csv(gap_rows, output_dir):
    if not gap_rows:
        return

    fieldnames = [
        "label",
        "run_name",
        "mode",
        "lr",
        "adapter_lr",
        "weight_decay",
        "warmup_pct",
        "batch_size",
        "epochs",
        "train_bps",
        "val_bps",
        "test_bps",
        "gap_bps",
        "train_nll",
        "val_nll",
        "test_nll",
        "gap_nll",
    ]
    path = output_dir / "generalisation_gap.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gap_rows)

    print("\n=== Generalisation Gap ===")
    print(
        f"{'Label':<26} {'mode':<8} {'train_bps':>10} {'val_bps':>8} "
        f"{'test_bps':>9} {'gap(test-train)':>16}"
    )
    print("-" * 92)
    for row in gap_rows:
        print(
            f"{row['label']:<26} {row['mode']:<8} {row['train_bps']:>10.4f} "
            f"{row['val_bps']:>8.4f} {row['test_bps']:>9.4f} "
            f"{row['gap_bps']:>16.4f}"
        )



def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--results-root", type=Path, default=Path("results_20ms"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    runs, labels = resolve_runs_and_labels(args)

    output_dir = args.output_dir
    if output_dir is None:
        if args.preset == "verified_combined":
            output_dir = Path("results_20ms/report/section3_generalisation_verified_combined")
        else:
            output_dir = Path("results_20ms/report/section4")

    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_data = []
    for run_dir, label in zip(runs, labels):
        rd = {
            "label": label,
            "run_name": run_dir.name,
            "history": load_history(run_dir),
            "metadata": load_metadata(run_dir),
            "split_summaries": load_split_summaries(run_dir),
        }
        rd["hparams"] = extract_hparams(rd["metadata"])
        runs_data.append(rd)
        if rd["split_summaries"] is None:
            print(
                f"[WARN] {label}: no split_eval_summary.json or "
                "combined_eval_summary.json found under artifacts"
            )

    plot_curves_overlay(
        runs_data,
        output_dir,
        "train_loss_per_masked_bin",
        "Loss per masked bin",
        "Training loss",
        "curves_train.png",
        plt,
    )
    plot_curves_overlay(
        runs_data,
        output_dir,
        "val_loss_per_masked_bin",
        "Loss per masked bin",
        "Validation loss",
        "curves_val.png",
        plt,
    )

    for rd in runs_data:
        plot_single_run_both(rd, output_dir, plt)

    gap_rows = collect_gap_data(runs_data)
    write_gap_csv(gap_rows, output_dir)

    for mode in ("default", "neuron", "causal"):
        if mode == "default":
            bps_title = "Generalisation across splits"
            nll_title = "Generalisation NLL across splits"
        else:
            bps_title = f"Generalisation across splits ({mode})"
            nll_title = f"Generalisation NLL across splits ({mode})"

        plot_split_lines(
            gap_rows,
            output_dir,
            mode=mode,
            metric="bps",
            ylabel="bits / spike vs per-neuron baseline",
            title=bps_title,
            fname=f"generalisation_bps_{mode}.png",
            plt=plt,
        )
        plot_split_lines(
            gap_rows,
            output_dir,
            mode=mode,
            metric="nll",
            ylabel="Model NLL per masked bin",
            title=nll_title,
            fname=f"generalisation_nll_{mode}.png",
            plt=plt,
        )

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
