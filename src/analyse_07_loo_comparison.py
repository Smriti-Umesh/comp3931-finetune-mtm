"""
LOO comparison across model families.

Scans per-run leave-one-neuron-out reports produced by `evaluate_loo.py`,


Outputs:
  - all_runs_summary.csv
  - selected_runs.csv
  - combined_summary.png
  - combined_bps_hist.png
  - combined_r2_hist.png
  - combined_r2_vs_rate.png
  - causal_summary.png
  - causal_bps_hist.png
  - causal_r2_hist.png
  - causal_r2_vs_rate.png
  - neuron_summary.png
  - neuron_bps_hist.png
  - neuron_r2_hist.png
  - neuron_r2_vs_rate.png

"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


FAMILY_SPECS = [
    {
        "group": "combined",
        "family_key": "ibl_mtm_combined",
        "label": "IBL-MtM transfer",
        "prefixes": ["ibl_mtm_combined_direct_full_"],
        "color": "#2F6BFF",
    },
    {
        "group": "combined",
        "family_key": "ndt1_direct_combined",
        "label": "NDT1 scratch",
        "prefixes": ["ndt1_direct_combined_full_"],
        "color": "#F28E2B",
    },
    {
        "group": "combined",
        "family_key": "ndt1_stitched_combined",
        "label": "Stitched NDT1 scratch",
        "prefixes": ["ndt1_stitched_combined_full_"],
        "color": "#59A14F",
    },
    {
        "group": "causal",
        "family_key": "ibl_mtm_causal",
        "label": "IBL-MtM transfer",
        "prefixes": ["causal_full_"],
        "color": "#2F6BFF",
    },
    {
        "group": "causal",
        "family_key": "ndt1_direct_causal",
        "label": "NDT1 scratch",
        "prefixes": ["ndt1_direct_causal_full_"],
        "color": "#F28E2B",
    },
    {
        "group": "causal",
        "family_key": "ndt1_stitched_causal",
        "label": "Stitched NDT1 scratch",
        "prefixes": ["ndt1_stitched_causal_full_"],
        "color": "#59A14F",
    },
    {
        "group": "neuron",
        "family_key": "ibl_mtm_neuron",
        "label": "IBL-MtM transfer",
        "prefixes": ["local_neuron_mask_full_"],
        "color": "#2F6BFF",
    },
    {
        "group": "neuron",
        "family_key": "ndt1_direct_neuron",
        "label": "NDT1 scratch",
        "prefixes": ["ndt1_direct_neuron_full_"],
        "color": "#F28E2B",
    },
    {
        "group": "neuron",
        "family_key": "ndt1_stitched_neuron",
        "label": "Stitched NDT1 scratch",
        "prefixes": ["ndt1_stitched_neuron_full_"],
        "color": "#59A14F",
    },
]

GROUP_TITLES = {
    "combined": "Combined-Objective Models",
    "causal": "Causal-Only Models",
    "neuron": "Neuron-Only Models",
}


def _style(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#D9D9D9", alpha=0.55, linewidth=0.8)


def _match_family(run_name: str) -> Optional[Dict[str, str]]:
    for spec in FAMILY_SPECS:
        if any(run_name.startswith(prefix) for prefix in spec["prefixes"]):
            return spec
    return None


def _summary_stats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    r2 = np.asarray([row["r2"] for row in rows if not math.isnan(row["r2"])], dtype=np.float64)
    bps = np.asarray([row["bps_vs_null"] for row in rows if not math.isnan(row["bps_vs_null"])], dtype=np.float64)
    rates = np.asarray([row["mean_rate"] for row in rows if not math.isnan(row["mean_rate"])], dtype=np.float64)
    return {
        "n_neurons": float(len(rows)),
        "median_r2": float(np.median(r2)) if r2.size else float("nan"),
        "mean_r2": float(np.mean(r2)) if r2.size else float("nan"),
        "frac_r2_positive": float(np.mean(r2 > 0)) if r2.size else float("nan"),
        "median_bps": float(np.median(bps)) if bps.size else float("nan"),
        "mean_bps": float(np.mean(bps)) if bps.size else float("nan"),
        "frac_bps_positive": float(np.mean(bps > 0)) if bps.size else float("nan"),
        "mean_rate_mean": float(np.mean(rates)) if rates.size else float("nan"),
    }


def load_loo_run(run_dir: Path) -> Optional[Dict[str, object]]:
    csv_path = run_dir / "loo_metrics.csv"
    if not csv_path.exists():
        return None

    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "neuron_idx": float(row["neuron_idx"]),
                "mean_rate": float(row["mean_rate"]),
                "total_spikes": float(row["total_spikes"]),
                "r2": float(row["r2"]),
                "window_corr": float(row["window_corr"]),
                "bps_vs_null": float(row["bps_vs_null"]),
            })

    spec = _match_family(run_dir.name)
    if spec is None:
        return None

    return {
        "run_name": run_dir.name,
        "run_dir": run_dir,
        "group": spec["group"],
        "family_key": spec["family_key"],
        "family_label": spec["label"],
        "color": spec["color"],
        "rows": rows,
        "summary": _summary_stats(rows),
    }


def _selection_score(run_payload: Dict[str, object], metric: str) -> float:
    return float(run_payload["summary"].get(metric, float("-inf")))


def load_runs(source_dir: Path, selection_metric: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    loaded: List[Dict[str, object]] = []
    for run_dir in sorted(source_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        payload = load_loo_run(run_dir)
        if payload is not None:
            loaded.append(payload)

    selected: List[Dict[str, object]] = []
    for spec in FAMILY_SPECS:
        family_runs = [run for run in loaded if run["family_key"] == spec["family_key"]]
        if not family_runs:
            continue
        best = max(
            family_runs,
            key=lambda run: (
                _selection_score(run, selection_metric),
                _selection_score(run, "median_r2"),
                _selection_score(run, "mean_bps"),
            ),
        )
        selected.append(best)

    return loaded, selected


def write_run_summaries(all_runs: List[Dict[str, object]], selected_runs: List[Dict[str, object]], output_dir: Path) -> None:
    all_rows = []
    for run in all_runs:
        row = {
            "run_name": run["run_name"],
            "group": run["group"],
            "family_key": run["family_key"],
            "family_label": run["family_label"],
        }
        row.update(run["summary"])
        all_rows.append(row)

    selected_names = {run["run_name"] for run in selected_runs}
    selected_rows = [row for row in all_rows if row["run_name"] in selected_names]

    for filename, rows in [
        ("all_runs_summary.csv", all_rows),
        ("selected_runs.csv", selected_rows),
    ]:
        if not rows:
            continue
        with open(output_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _comparison_runs(selected_runs: List[Dict[str, object]], group: str) -> List[Dict[str, object]]:
    return [run for run in selected_runs if run["group"] == group]


def plot_summary_bar(runs: List[Dict[str, object]], title_prefix: str, output_path: Path, plt) -> None:
    labels = [run["family_label"] for run in runs]
    colors = [run["color"] for run in runs]
    median_bps = np.asarray([run["summary"]["median_bps"] for run in runs], dtype=np.float64)
    median_r2 = np.asarray([run["summary"]["median_r2"] for run in runs], dtype=np.float64)
    frac_pos = np.asarray([run["summary"]["frac_r2_positive"] for run in runs], dtype=np.float64)
    metrics = [
        ("Median LOO bps", median_bps, "{:.3f}"),
        ("Median LOO R^2", median_r2, "{:.3f}"),
        ("Fraction R^2 > 0", frac_pos, "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    x = np.arange(len(runs))
    for ax, (title, values, fmt) in zip(axes, metrics):
        bars = ax.bar(x, values, color=colors, width=0.62)
        if "Fraction" in title:
            ax.set_ylim(0.0, 1.0)
        elif np.nanmin(values) < 0:
            ax.axhline(0.0, color="#3B3B3B", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        _style(ax)
        for bar, value in zip(bars, values):
            if math.isnan(float(value)):
                continue
            y = float(value)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                fmt.format(float(value)),
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle(f"{title_prefix}: Summary", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pooled_metric(runs: List[Dict[str, object]], key: str) -> np.ndarray:
    values = []
    for run in runs:
        values.extend([row[key] for row in run["rows"] if not math.isnan(row[key])])
    return np.asarray(values, dtype=np.float64)


def plot_overlay_hist(
    runs: List[Dict[str, object]],
    key: str,
    xlabel: str,
    title: str,
    output_path: Path,
    plt,
) -> None:
    pooled = _pooled_metric(runs, key)
    if pooled.size == 0:
        return
    bins = np.histogram_bin_edges(pooled, bins=32)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for run in runs:
        values = np.asarray([row[key] for row in run["rows"] if not math.isnan(row[key])], dtype=np.float64)
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.28,
            color=run["color"],
            edgecolor=run["color"],
            linewidth=1.0,
            label=run["family_label"],
        )
        ax.axvline(np.median(values), color=run["color"], linewidth=1.8, linestyle="--")

    ax.axvline(0.0, color="#3B3B3B", linewidth=1.0, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Neuron count")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rate_scatter_panels(runs: List[Dict[str, object]], title: str, output_path: Path, plt) -> None:
    all_rates = np.asarray(
        [row["mean_rate"] for run in runs for row in run["rows"] if not math.isnan(row["mean_rate"])],
        dtype=np.float64,
    )
    all_r2 = np.asarray(
        [row["r2"] for run in runs for row in run["rows"] if not math.isnan(row["r2"])],
        dtype=np.float64,
    )
    if all_rates.size == 0 or all_r2.size == 0:
        return

    x_min, x_max = float(np.nanmin(all_rates)), float(np.nanmax(all_rates))
    y_min = min(-0.05, float(np.nanmin(all_r2)))
    y_max = max(0.05, float(np.nanmax(all_r2)))

    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.5), sharey=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        rows = [row for row in run["rows"] if not math.isnan(row["r2"])]
        rates = np.asarray([row["mean_rate"] for row in rows], dtype=np.float64)
        r2 = np.asarray([row["r2"] for row in rows], dtype=np.float64)
        ax.scatter(
            rates,
            r2,
            s=20,
            color=run["color"],
            alpha=0.68,
            edgecolors="white",
            linewidths=0.25,
        )
        ax.axhline(0.0, color="#3B3B3B", linewidth=1.0, linestyle="--")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Mean firing rate (spk/bin)")
        ax.set_title(run["family_label"], fontsize=10)
        _style(ax, grid_axis="both")

    axes[0].set_ylabel("LOO R^2")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_group_comparison(selected_runs: List[Dict[str, object]], group: str, output_dir: Path, plt) -> None:
    runs = _comparison_runs(selected_runs, group)
    if len(runs) != 3:
        print(f"[WARN] skipping {group}: expected 3 selected families, found {len(runs)}")
        return

    title_prefix = GROUP_TITLES[group]
    plot_summary_bar(runs, title_prefix, output_dir / f"{group}_summary.png", plt)
    plot_overlay_hist(
        runs,
        key="bps_vs_null",
        xlabel="LOO bits / spike",
        title=f"{title_prefix}: LOO Bits per Spike",
        output_path=output_dir / f"{group}_bps_hist.png",
        plt=plt,
    )
    plot_overlay_hist(
        runs,
        key="r2",
        xlabel="LOO R^2",
        title=f"{title_prefix}: LOO R^2",
        output_path=output_dir / f"{group}_r2_hist.png",
        plt=plt,
    )
    plot_rate_scatter_panels(
        runs,
        title=f"{title_prefix}: LOO R^2 vs Mean Firing Rate",
        output_path=output_dir / f"{group}_r2_vs_rate.png",
        plt=plt,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path, default=Path("results_20ms/report/loo"))
    ap.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/loo_comparison"))
    ap.add_argument(
        "--selection-metric",
        choices=["median_bps", "mean_bps", "median_r2", "mean_r2"],
        default="median_bps",
        help="Metric used to select the representative run within each family",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {args.source_dir}")

    all_runs, selected_runs = load_runs(args.source_dir, args.selection_metric)
    if not all_runs:
        raise SystemExit(f"No loo_metrics.csv files found under {args.source_dir}")

    write_run_summaries(all_runs, selected_runs, args.output_dir)
    render_group_comparison(selected_runs, "combined", args.output_dir, plt)
    render_group_comparison(selected_runs, "causal", args.output_dir, plt)
    render_group_comparison(selected_runs, "neuron", args.output_dir, plt)

    print("\nSelected runs")
    for run in selected_runs:
        summary = run["summary"]
        print(
            f"{run['group']:<8} {run['family_label']:<24} "
            f"{run['run_name']:<55} "
            f"median_bps={summary['median_bps']:+.4f} "
            f"median_r2={summary['median_r2']:+.4f}"
        )

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
