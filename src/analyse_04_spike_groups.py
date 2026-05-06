"""
Spike Group Analysis

Breaks the held-out loss down by the true spike count in each bin

Loads eval_predictions.npz (one or more runs) and produces:
  - spike_groups_<label>.png  : per-run grouped bar chart (model vs baseline)
  - spike_groups_gain.png     : NLL gain comparison across all conditions
  - spike_groups.csv          : raw numbers for all runs

"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def load_predictions(run_dir: Path, masking_mode: str = "neuron"):
    for candidate in [
        run_dir / "artifacts" / "eval_predictions.npz",
        run_dir / "artifacts" / f"eval_test_{masking_mode}" / "eval_predictions.npz",
        run_dir / "eval_predictions.npz",
    ]:
        if candidate.exists():
            return np.load(candidate)
    raise FileNotFoundError(f"eval_predictions.npz not found under {run_dir}")



GROUPS = [
    ("0 spikes",  lambda t: t == 0),
    ("1 spike",   lambda t: t == 1),
    ("2+ spikes", lambda t: t >= 2),
    ("3+ spikes", lambda t: t >= 3),
    ("5+ spikes", lambda t: t >= 5),
]


def poisson_nll(rate, target, eps=1e-9):
    r = np.maximum(rate, eps)
    return r - target * np.log(r)


def compute_groups(preds):
    targets    = preds["targets"].astype(np.float64)    # [W, T, N]
    pred_rates = preds["pred_rates"].astype(np.float64) # [W, T, N]
    per_neuron = preds["baseline_per_neuron"].astype(np.float64)  # [N]
    mask_bool  = preds["eval_mask"].astype(bool)        # [W, T, N]

    baseline_rates = np.broadcast_to(per_neuron[None, None, :], targets.shape)
    model_nll    = poisson_nll(pred_rates, targets)
    baseline_nll = poisson_nll(baseline_rates, targets)

    rows = []
    for label, cond_fn in GROUPS:
        keep = mask_bool & cond_fn(targets)
        if not np.any(keep):
            continue
        rows.append({
            "group":        label,
            "bins":         int(np.sum(keep)),
            "spikes":       float(np.sum(targets[keep])),
            "model_nll":    float(np.mean(model_nll[keep])),
            "baseline_nll": float(np.mean(baseline_nll[keep])),
            "nll_gain":     float(np.mean(baseline_nll[keep] - model_nll[keep])),
            "pred_mean":    float(np.mean(pred_rates[keep])),
            "target_mean":  float(np.mean(targets[keep])),
        })
    return rows



def print_groups(rows, label):
    print(f"\n=== Spike-count groups: {label} ===")
    print(f"{'Group':<12} {'bins':>8} {'model_nll':>10} {'baseline':>10} {'gain':>8}")
    print("-" * 52)
    for r in rows:
        print(f"{r['group']:<12} {r['bins']:>8} {r['model_nll']:>10.4f} "
              f"{r['baseline_nll']:>10.4f} {r['nll_gain']:>+8.4f}")
    # Flag: is gain positive for actual spike events?
    for r in rows:
        if r["group"] == "1 spike":
            sign = "✓ GAIN" if r["nll_gain"] > 0 else "✗ NO GAIN"
            print(f"  → 1-spike bins: {sign} ({r['nll_gain']:+.4f})")


def write_csv(all_rows, output_dir):
    flat = []
    for label, rows in all_rows:
        for r in rows:
            flat.append({"label": label, **r})
    path = output_dir / "spike_groups.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)


def plot_single_run(rows, output_dir, label, plt):
    """Model vs baseline NLL per spike-count group for one run."""
    group_labels = [r["group"] for r in rows]
    model_nll    = np.array([r["model_nll"]    for r in rows])
    baseline_nll = np.array([r["baseline_nll"] for r in rows])
    gain         = np.array([r["nll_gain"]      for r in rows])
    x     = np.arange(len(rows))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].bar(x - width/2, model_nll,    width=width, label="model")
    axes[0].bar(x + width/2, baseline_nll, width=width, label="per-neuron baseline")
    axes[0].set_ylabel("NLL per masked bin")
    axes[0].set_title("Model vs baseline NLL by spike count")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(x); axes[0].set_xticklabels(group_labels, rotation=20, ha="right")
    _style(axes[0])

    bar_colors = ["green" if g > 0 else "tomato" for g in gain]
    axes[1].bar(x, gain, color=bar_colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Baseline − model NLL  (positive = model wins)")
    axes[1].set_title("NLL gain by spike count")
    axes[1].set_xticks(x); axes[1].set_xticklabels(group_labels, rotation=20, ha="right")
    _style(axes[1])

    slug = label.replace(" ", "_").replace("/", "-")
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / f"spike_groups_{slug}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain_comparison(all_rows, output_dir, plt):
    """Overlay the NLL gain across runs for each spike-count group."""
    # Collect all group labels (may differ if some groups are empty)
    group_set = []
    for _, rows in all_rows:
        for r in rows:
            if r["group"] not in group_set:
                group_set.append(r["group"])

    n_groups = len(group_set)
    n_runs   = len(all_rows)
    x        = np.arange(n_groups)
    width    = 0.8 / n_runs

    fig, ax = plt.subplots(figsize=(max(8.0, 2.0 * n_groups), 4.8))
    for i, (label, rows) in enumerate(all_rows):
        gains = {r["group"]: r["nll_gain"] for r in rows}
        vals  = [gains.get(g, float("nan")) for g in group_set]
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(group_set, rotation=20, ha="right")
    ax.set_ylabel("Baseline - model NLL  (positive = model wins)")
    ax.set_title("Spike-event gain comparison across conditions")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "spike_groups_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs",       nargs="+", required=True, type=Path)
    ap.add_argument("--labels",     nargs="+", default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/spike_grp_analysis"))
    ap.add_argument(
        "--masking-mode",
        default="neuron",
        choices=["neuron", "causal"],
        help="Which nested eval_test_<mode> directory to use for combined runs. "
             "Ignored for single-task runs with flat artifacts/eval_predictions.npz.",
    )
    args = ap.parse_args()

    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must have the same count as --runs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    for run_dir, label in zip(args.runs, labels):
        preds = load_predictions(run_dir, masking_mode=args.masking_mode)
        rows  = compute_groups(preds)
        print_groups(rows, label)
        all_rows.append((label, rows))
        plot_single_run(rows, args.output_dir, label, plt)
        preds.close()

    write_csv(all_rows, args.output_dir)
    plot_gain_comparison(all_rows, args.output_dir, plt)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
