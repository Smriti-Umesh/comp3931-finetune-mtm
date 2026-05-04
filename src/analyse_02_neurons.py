"""
 Per-neuron firing-rate prediction quality.

The core question: for each held-out neuron, how well does the model infer its
activity from the rest of the population across test windows?

Loads eval_predictions.npz and produces:
  - neuron_window_r2_hist.png    : distribution of per-neuron window-level R²
  - neuron_r2_vs_rate.png        : R² vs mean firing rate scatter
  - neuron_traces_top.png        : true vs predicted traces for best-predicted neurons
  - neuron_traces_bottom.png     : true vs predicted traces for worst-predicted neurons
  - neuron_metrics.csv           : per-neuron R², correlation, mean rate


"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np



def load_predictions(run_dir: Path):
    for candidate in [run_dir / "artifacts" / "eval_predictions.npz",
                      run_dir / "eval_predictions.npz"]:
        if candidate.exists():
            return np.load(candidate)
    raise FileNotFoundError(f"eval_predictions.npz not found under {run_dir}")


def window_r2_per_neuron(targets, pred_rates, heldout_indices):
    """
    For each held-out neuron, average over the 100 time bins to get one
    activity value per test window, then compute R^2 between true and predicted.

    targets:          [W, T, N]  true spike counts
    pred_rates:       [W, T, N]  model predicted rates
    heldout_indices:  [K]        which neuron columns were actually held out

    Returns a dict keyed by neuron index with:
        r2       : scalar  (window-level R^2)
        corr     : scalar  (Pearson r, window-level)
        mean_rate: scalar  (mean spikes/bin across all windows x time)
    """
    W, T, N = targets.shape
    results = {}

    for n in heldout_indices:
        true_win  = targets[:, :, n].mean(axis=1)   # [W] — mean spikes per window
        pred_win  = pred_rates[:, :, n].mean(axis=1) # [W] — mean predicted rate per window
        mean_rate = float(targets[:, :, n].mean())

        # R^2 = 1 - SS_res / SS_tot
        ss_res = float(np.sum((true_win - pred_win) ** 2))
        ss_tot = float(np.sum((true_win - true_win.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

        # Pearson correlation (window-level)
        if true_win.std() > 1e-9 and pred_win.std() > 1e-9:
            corr = float(np.corrcoef(true_win, pred_win)[0, 1])
        else:
            corr = float("nan")

        results[int(n)] = {"r2": r2, "corr": corr, "mean_rate": mean_rate,
                           "true_win": true_win, "pred_win": pred_win}

    return results


def write_csv(results, output_dir, label):
    rows = [{"neuron_idx": n,
             "window_r2":  v["r2"],
             "window_corr": v["corr"],
             "mean_rate":   v["mean_rate"]}
            for n, v in sorted(results.items())]
    path = output_dir / "neuron_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print summary stats
    r2s = np.array([v["r2"] for v in results.values()])
    valid = r2s[~np.isnan(r2s)]
    print(f"\n=== Per-neuron window-level R² ({label}) ===")
    print(f"  neurons evaluated : {len(valid)}")
    print(f"  median R^2         : {np.median(valid):.4f}")
    print(f"  mean R^2           : {np.mean(valid):.4f}")
    print(f"  fraction R^2 > 0   : {np.mean(valid > 0)*100:.1f}%")
    print(f"  p10 / p90         : {np.percentile(valid, 10):.4f} / {np.percentile(valid, 90):.4f}")




def plot_r2_histogram(results, output_dir, label, plt):
    r2s = np.array([v["r2"] for v in results.values()])
    valid = r2s[~np.isnan(r2s)]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(valid, bins=30, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0,            color="black",  linewidth=1.2, linestyle="--", label="R²=0 (no gain)")
    ax.axvline(float(np.median(valid)), color="red", linewidth=1.2, linestyle="-",
               label=f"median={np.median(valid):.3f}")
    ax.set_xlabel("Window-level R² (held-out neurons)")
    ax.set_ylabel("Neuron count")
    ax.set_title(f"Per-neuron co-smoothing quality\n{label}")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "neuron_window_r2_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r2_vs_rate(results, output_dir, label, plt):
    r2s   = np.array([v["r2"]        for v in results.values()])
    rates = np.array([v["mean_rate"] for v in results.values()])
    valid = ~np.isnan(r2s)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sc = ax.scatter(rates[valid], r2s[valid], s=18, alpha=0.6,
                    c=r2s[valid], cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Mean firing rate (spikes / bin, training set)")
    ax.set_ylabel("Window-level R²")
    ax.set_title(f"Prediction quality vs firing rate\n{label}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(sc, ax=ax, label="R²")
    fig.tight_layout()
    fig.savefig(output_dir / "neuron_r2_vs_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_neuron_traces(results, output_dir, label, n_show, best, plt):
    """Show window-mean true spike count vs predicted rate across test windows."""
    r2s      = {n: v["r2"] for n, v in results.items() if not np.isnan(v["r2"])}
    sorted_n = sorted(r2s, key=lambda n: r2s[n], reverse=best)[:n_show]

    n_cols = min(n_show, 3)
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    window_idx = np.arange(list(results.values())[0]["true_win"].shape[0])

    for i, n in enumerate(sorted_n):
        ax  = axes[i // n_cols][i % n_cols]
        v   = results[n]
        ax.plot(window_idx, v["true_win"],  label="true",      linewidth=1.5)
        ax.plot(window_idx, v["pred_win"],  label="predicted",  linewidth=1.5, linestyle="--")
        ax.set_title(f"Neuron {n} | R²={v['r2']:.3f} | rate={v['mean_rate']:.4f}",
                     fontsize=9)
        ax.set_xlabel("test window index", fontsize=8)
        ax.set_ylabel("mean spikes/bin",   fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)

  
    for i in range(len(sorted_n), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    quality = "best" if best else "worst"
    fig.suptitle(f"{label}: {quality} {n_show} held-out neurons by window-level R²", fontsize=11)
    fig.tight_layout()
    fname = f"neuron_traces_{quality}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",        required=True, type=Path)
    ap.add_argument("--label",      default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/section2"))
    ap.add_argument("--n-traces",   type=int, default=6,
                    help="How many best/worst neurons to show in trace plots")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = args.label or args.run.name
    preds = load_predictions(args.run)

    targets         = preds["targets"].astype(np.float64)       # [W, T, N]
    pred_rates      = preds["pred_rates"].astype(np.float64)    # [W, T, N]
    heldout_indices = preds["heldout_neuron_indices"]            # [K]

    if heldout_indices.size == 0:
        raise SystemExit("No held-out neuron indices found in eval_predictions.npz. "
                         "This run used causal masking — use the causal variant of this script.")

    results = window_r2_per_neuron(targets, pred_rates, heldout_indices)

    write_csv(results, args.output_dir, label)
    plot_r2_histogram(results, args.output_dir, label, plt)
    plot_r2_vs_rate(results, args.output_dir, label, plt)
    plot_neuron_traces(results, args.output_dir, label, args.n_traces, best=True,  plt=plt)
    plot_neuron_traces(results, args.output_dir, label, args.n_traces, best=False, plt=plt)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
