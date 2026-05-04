"""
 Selected latent-space comparisons across representative runs.

This script picks a compact, thesis-friendly set of runs and fits a shared
latent embedding for each comparison family:

  - combined_neuron : best 8BS MtM, best true015 MtM, best rerun MtM,
                      best NDT1 scratch combined, best stitched combined
                      (using eval_test_neuron latents)
  - combined_causal : same selected combined runs, but using eval_test_causal
                      latents
  - neuron_family   : best MtM neuron, best NDT1 scratch neuron,
                      best stitched neuron
  - causal_family   : best MtM causal, best NDT1 scratch causal,
                      best stitched causal

Selection rules:
  - combined MtM pools are ranked by average test BPS across neuron + causal
  - single-task pools are ranked by test BPS

For each comparison family, the script:
  - loads mean-pooled unmasked latents [N, 512]
  - balances the number of windows per run
  - optionally L2 normalises the latent vectors
  - fits a shared PCA basis and projects to a configurable PCA subspace
  - fits shared 2D PCA / UMAP / t-SNE embeddings
  - saves clean plots coloured by run label and by mean population firing rate
  - writes selection and point-level CSVs for provenance
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

# Note: experiment names can change depending on run names
# The names originally went ibl_mtm_*_lr2e-5_JOB_ID 
# But the JOB ID has been removed and this set of file names should be changed
# Depending on the names/job IDs new runs produce
SELECTION_POOLS = {
    "combined_8bs": {
        "kind": "combined",
        "metric": "avg_test_bps",
        "label": "MtM 8BS",
        "runs": [
            "ibl_mtm_combined_direct_full_plain_lr2e-5",
            "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5",
        ],
    },
    "combined_true015": {
        "kind": "combined",
        "metric": "avg_test_bps",
        "label": "MtM best true015",
        "runs": [
            "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd001",
            "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd01",
            "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd001",
            "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd01",
        ],
    },
    "combined_rerun": {
        "kind": "combined",
        "metric": "avg_test_bps",
        "label": "MtM best rerun",
        "runs": [
            "ibl_mtm_combined_direct_full_lr1e-5_adapter5e-5_rerun",
            "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun",
            "ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun",
            "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun",
            "ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun",
        ],
    },
    "combined_scratch": {
        "kind": "combined",
        "metric": "avg_test_bps",
        "label": "NDT1 scratch",
        "runs": [
            "ndt1_direct_combined_full_lr2e-5",
        ],
    },
    "combined_stitched": {
        "kind": "combined",
        "metric": "avg_test_bps",
        "label": "Stitched NDT1",
        "runs": [
            "ndt1_stitched_combined_full_prompt_lr2e-5",
        ],
    },
    "neuron_mtm": {
        "kind": "single",
        "metric": "test_bps",
        "label": "MtM best neuron",
        "runs": [
            "local_neuron_mask_full_plain_lr2e-5",
            "local_neuron_mask_full_lr2e-5_adapter5e-5",
            "local_neuron_mask_full_lr2e-5_adapter5e-5,
        ],
    },
    "neuron_scratch": {
        "kind": "single",
        "metric": "test_bps",
        "label": "NDT1 scratch neuron",
        "runs": [
            "ndt1_direct_neuron_full_lr2e-5",
        ],
    },
    "neuron_stitched": {
        "kind": "single",
        "metric": "test_bps",
        "label": "Stitched NDT1 neuron",
        "runs": [
            "ndt1_stitched_neuron_full_prompt_lr2e-5",
        ],
    },
    "causal_mtm": {
        "kind": "single",
        "metric": "test_bps",
        "label": "MtM best causal",
        "runs": [
            "causal_full_plain_lr2e-5",
            "causal_full_lr2e-5_adapter5e-5",
            "causal_full_lr2e-5_adapter5e-5",
            "causal_full_lr2e-5_adapter5e-5",
        ],
    },
    "causal_scratch": {
        "kind": "single",
        "metric": "test_bps",
        "label": "NDT1 scratch causal",
        "runs": [
            "ndt1_direct_causal_full_lr2e-5",
        ],
    },
    "causal_stitched": {
        "kind": "single",
        "metric": "test_bps",
        "label": "Stitched NDT1 causal",
        "runs": [
            "ndt1_stitched_causal_full_prompt_lr2e-5",
        ],
    },
}


GROUPS = {
    "combined_neuron": [
        "combined_8bs",
        "combined_true015",
        "combined_rerun",
        "combined_scratch",
        "combined_stitched",
    ],
    "combined_causal": [
        "combined_8bs",
        "combined_true015",
        "combined_rerun",
        "combined_scratch",
        "combined_stitched",
    ],
    "neuron_family": [
        "neuron_mtm",
        "neuron_scratch",
        "neuron_stitched",
    ],
    "causal_family": [
        "causal_mtm",
        "causal_scratch",
        "causal_stitched",
    ],
}


GROUP_TITLES = {
    "combined_neuron": "Combined models: neuron-eval latent comparison",
    "combined_causal": "Combined models: causal-eval latent comparison",
    "neuron_family": "Neuron-trained models: latent comparison",
    "causal_family": "Causal-trained models: latent comparison",
}


PALETTE = [
    "#0f766e",
    "#f59e0b",
    "#dc2626",
    "#2563eb",
    "#65a30d",
    "#7c3aed",
]


def _artifact_base(run_dir: Path, mode: str | None):
    if mode in {"neuron", "causal"}:
        nested = run_dir / "artifacts" / f"eval_test_{mode}"
        if nested.exists():
            return nested
    return run_dir / "artifacts"


def load_latents_and_rates(run_dir: Path, mode: str | None):
    base = _artifact_base(run_dir, mode)
    latent_path = base / "unmasked_latent_mean_pooled.npy"
    pred_path = base / "eval_predictions.npz"
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing latent file: {latent_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    latents = np.load(latent_path).astype(np.float64)
    with np.load(pred_path) as preds:
        targets = preds["targets"].astype(np.float64)
    mean_rates = targets.mean(axis=(1, 2))
    return latents, mean_rates


def load_metadata(run_dir: Path):
    path = run_dir / "artifacts" / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def extract_hparams(metadata):
    args = metadata.get("args", {})
    config = metadata.get("config", {})
    optimizer = config.get("optimizer", {})
    return {
        "lr": args.get("lr", optimizer.get("lr")),
        "adapter_lr": args.get("adapter_lr"),
        "weight_decay": args.get("weight_decay", optimizer.get("wd")),
        "warmup_pct": args.get("warmup_pct", optimizer.get("warmup_pct")),
        "batch_size": args.get("batch_size"),
        "epochs": args.get("epochs"),
    }


def load_single_metric(run_dir: Path):
    path = run_dir / "artifacts" / "eval_metrics.json"
    data = json.loads(path.read_text())
    return {
        "score": float(data["bits_per_spike_vs_per_neuron_mean"]),
        "test_bps": float(data["bits_per_spike_vs_per_neuron_mean"]),
        "test_r2": float(data.get("median_window_r2", float("nan"))),
    }


def load_combined_metric(run_dir: Path):
    path = run_dir / "artifacts" / "combined_eval_summary.json"
    data = json.loads(path.read_text())
    neuron_bps = float(data["neuron"]["test"]["bits_per_spike_vs_per_neuron_mean"])
    causal_bps = float(data["causal"]["test"]["bits_per_spike_vs_per_neuron_mean"])
    return {
        "score": (neuron_bps + causal_bps) / 2.0,
        "neuron_test_bps": neuron_bps,
        "causal_test_bps": causal_bps,
    }


def select_best_runs(results_dir: Path):
    selections = {}
    for key, spec in SELECTION_POOLS.items():
        candidates = []
        for run_name in spec["runs"]:
            run_dir = results_dir / run_name
            if spec["kind"] == "combined":
                metrics = load_combined_metric(run_dir)
            else:
                metrics = load_single_metric(run_dir)
            candidates.append((metrics["score"], run_name, metrics))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        score, run_name, metrics = candidates[0]
        run_dir = results_dir / run_name
        metadata = load_metadata(run_dir)
        selections[key] = {
            "selection_key": key,
            "display_label": spec["label"],
            "run_name": run_name,
            "run_dir": run_dir,
            "selection_metric": spec["metric"],
            "selection_score": score,
            "kind": spec["kind"],
            **metrics,
            **extract_hparams(metadata),
        }
    return selections


def balanced_subsample(items, max_points_per_run, random_seed):
    min_count = min(item["latents"].shape[0] for item in items)
    n_points = min_count if max_points_per_run is None else min(min_count, max_points_per_run)
    rng = np.random.default_rng(random_seed)

    sampled = []
    for item in items:
        n_total = item["latents"].shape[0]
        if n_points < n_total:
            indices = np.sort(rng.choice(n_total, size=n_points, replace=False))
        else:
            indices = np.arange(n_total)
        sampled.append(
            {
                **item,
                "latents": item["latents"][indices],
                "mean_rates": item["mean_rates"][indices],
                "window_index": indices.astype(int),
            }
        )
    return sampled


def maybe_l2_normalise(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return X / norms


def run_pca(X, n_components):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U * s
    explained = (s ** 2)
    explained = explained / max(explained.sum(), 1e-12)
    n_components = min(n_components, coords.shape[1])
    return coords[:, :n_components], explained[:n_components], mean, Vt[:n_components]


def run_umap(X, random_state, n_neighbors=15, min_dist=0.1):
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", Path(tempfile.gettempdir()) / "numba_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    try:
        import umap
    except ModuleNotFoundError:
        return None, "umap-learn is not installed"

    n_samples = X.shape[0]
    if n_samples < 4:
        return None, "Need at least 4 samples for UMAP"
    n_neighbors = max(2, min(int(n_neighbors), n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X), None


def run_tsne(X, random_state, perplexity=30.0):
    n_samples = X.shape[0]
    if n_samples < 4:
        return None, "Need at least 4 samples for t-SNE"
    valid_perplexity = max(2.0, min(float(perplexity), float(n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=valid_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X), None


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15)


def _scatter_by_run(ax, coords, point_rows, color_map, title):
    labels = []
    for label in dict.fromkeys(row["label"] for row in point_rows):
        labels.append(label)
    for label in labels:
        idx = [i for i, row in enumerate(point_rows) if row["label"] == label]
        pts = coords[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            alpha=0.8,
            color=color_map[label],
            label=label,
        )
    ax.set_title(title)
    _style(ax)


def _scatter_by_rate(ax, coords, rates, title, plt):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=rates, cmap="viridis", s=18, alpha=0.85)
    ax.set_title(title)
    _style(ax)
    plt.colorbar(sc, ax=ax, label="Mean population firing rate", shrink=0.85)


def plot_embedding_pair(coords, point_rows, rates, color_map, title_prefix, out_path, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    _scatter_by_run(axes[0], coords, point_rows, color_map, f"{title_prefix} coloured by run")
    _scatter_by_rate(axes[1], coords, rates, f"{title_prefix} coloured by mean rate", plt)
    axes[0].legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overview(pca2, umap2, tsne2, point_rows, rates, color_map, title, out_path, plt):
    fig, axes = plt.subplots(3, 2, figsize=(13.8, 15.8))
    embedding_specs = [
        ("Shared PCA", pca2),
        ("Shared UMAP", umap2),
        ("Shared t-SNE", tsne2),
    ]
    for row_idx, (name, coords) in enumerate(embedding_specs):
        _scatter_by_run(axes[row_idx, 0], coords, point_rows, color_map, f"{name} coloured by run")
        _scatter_by_rate(axes[row_idx, 1], coords, rates, f"{name} coloured by mean rate", plt)
        if row_idx == 0:
            axes[row_idx, 0].legend(
                frameon=False,
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
            )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(requested: Path, overwrite: bool):
    requested = requested.resolve()
    if overwrite:
        requested.mkdir(parents=True, exist_ok=True)
        return requested

    if (not requested.exists()) or (requested.exists() and not any(requested.iterdir())):
        requested.mkdir(parents=True, exist_ok=True)
        return requested

    parent = requested.parent
    stem = requested.name
    idx = 2
    while True:
        candidate = parent / f"{stem}_v{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            print(f"[INFO] output dir exists, writing to: {candidate}")
            return candidate
        idx += 1


def build_group_items(group_name, selections):
    mode = None
    if group_name == "combined_neuron":
        mode = "neuron"
    elif group_name == "combined_causal":
        mode = "causal"

    items = []
    for selection_key in GROUPS[group_name]:
        sel = selections[selection_key]
        latents, mean_rates = load_latents_and_rates(sel["run_dir"], mode=mode)
        items.append(
            {
                "selection_key": selection_key,
                "label": sel["display_label"],
                "run_name": sel["run_name"],
                "run_dir": sel["run_dir"],
                "mode": mode or group_name.split("_")[0],
                "latents": latents,
                "mean_rates": mean_rates,
                "selection_metric": sel["selection_metric"],
                "selection_score": sel["selection_score"],
            }
        )
    return items


def render_group(group_name, items, output_dir: Path, args, plt):
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    sampled = balanced_subsample(items, args.max_points_per_run, args.random_seed)

    point_rows = []
    X_parts = []
    rates_parts = []
    for item in sampled:
        X = item["latents"]
        if args.l2_normalise:
            X = maybe_l2_normalise(X)
        X_parts.append(X)
        rates_parts.append(item["mean_rates"])
        for local_idx, (window_idx, mean_rate) in enumerate(zip(item["window_index"], item["mean_rates"])):
            point_rows.append(
                {
                    "group": group_name,
                    "label": item["label"],
                    "run_name": item["run_name"],
                    "window_index": int(window_idx),
                    "mean_rate": float(mean_rate),
                    "selection_metric": item["selection_metric"],
                    "selection_score": float(item["selection_score"]),
                    "local_index": int(local_idx),
                }
            )

    X_all = np.concatenate(X_parts, axis=0)
    rates_all = np.concatenate(rates_parts, axis=0)
    pca_coords, explained, _, _ = run_pca(X_all, args.pca_dims)
    pca_2d = pca_coords[:, :2]

    umap_coords, umap_error = run_umap(
        pca_coords,
        random_state=args.random_seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    tsne_coords, tsne_error = run_tsne(
        pca_coords,
        random_state=args.random_seed,
        perplexity=args.tsne_perplexity,
    )

    if umap_coords is None:
        print(f"[WARN] {group_name}: skipping UMAP: {umap_error}")
        umap_coords = np.full((X_all.shape[0], 2), np.nan)
    if tsne_coords is None:
        print(f"[WARN] {group_name}: skipping t-SNE: {tsne_error}")
        tsne_coords = np.full((X_all.shape[0], 2), np.nan)

    unique_labels = list(dict.fromkeys(row["label"] for row in point_rows))
    color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(unique_labels)}

    for row, pca_xy, umap_xy, tsne_xy in zip(point_rows, pca_2d, umap_coords, tsne_coords):
        row["pca1"] = float(pca_xy[0])
        row["pca2"] = float(pca_xy[1])
        row["umap1"] = float(umap_xy[0]) if np.isfinite(umap_xy[0]) else float("nan")
        row["umap2"] = float(umap_xy[1]) if np.isfinite(umap_xy[1]) else float("nan")
        row["tsne1"] = float(tsne_xy[0]) if np.isfinite(tsne_xy[0]) else float("nan")
        row["tsne2"] = float(tsne_xy[1]) if np.isfinite(tsne_xy[1]) else float("nan")

    write_csv(group_dir / "embedding_points.csv", point_rows)

    summary_rows = []
    for item in sampled:
        summary_rows.append(
            {
                "label": item["label"],
                "run_name": item["run_name"],
                "mode": item["mode"],
                "n_windows_used": item["latents"].shape[0],
                "selection_metric": item["selection_metric"],
                "selection_score": item["selection_score"],
            }
        )
    write_csv(group_dir / "selected_runs.csv", summary_rows)

    plot_embedding_pair(
        pca_2d,
        point_rows,
        rates_all,
        color_map,
        "Shared PCA",
        group_dir / "latent_pca_shared.png",
        plt,
    )
    plot_embedding_pair(
        umap_coords,
        point_rows,
        rates_all,
        color_map,
        "Shared UMAP",
        group_dir / "latent_umap_shared.png",
        plt,
    )
    plot_embedding_pair(
        tsne_coords,
        point_rows,
        rates_all,
        color_map,
        "Shared t-SNE",
        group_dir / "latent_tsne_shared.png",
        plt,
    )
    title = GROUP_TITLES[group_name]
    if args.l2_normalise:
        title += " (L2-normalised)"
    plot_overview(
        pca_2d,
        umap_coords,
        tsne_coords,
        point_rows,
        rates_all,
        color_map,
        title,
        group_dir / "latent_overview.png",
        plt,
    )

    scree_rows = []
    for idx, value in enumerate(explained[: min(20, len(explained))], start=1):
        scree_rows.append({"pc": idx, "fraction_variance_explained": float(value)})
    write_csv(group_dir / "pca_scree.csv", scree_rows)

    print(f"[GROUP] {group_name} -> {group_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_20ms"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_20ms/report/latent_selected_comparison"),
    )
    parser.add_argument("--pca-dims", type=int, default=40)
    parser.add_argument("--max-points-per-run", type=int, default=None)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--l2-normalise", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selections = select_best_runs(args.results_dir)
    selection_rows = []
    for selection_key, sel in selections.items():
        selection_rows.append(
            {
                "selection_key": selection_key,
                "display_label": sel["display_label"],
                "run_name": sel["run_name"],
                "selection_metric": sel["selection_metric"],
                "selection_score": sel["selection_score"],
                "kind": sel["kind"],
                "lr": sel.get("lr"),
                "adapter_lr": sel.get("adapter_lr"),
                "weight_decay": sel.get("weight_decay"),
                "warmup_pct": sel.get("warmup_pct"),
                "batch_size": sel.get("batch_size"),
                "epochs": sel.get("epochs"),
                "neuron_test_bps": sel.get("neuron_test_bps"),
                "causal_test_bps": sel.get("causal_test_bps"),
                "test_bps": sel.get("test_bps"),
                "test_r2": sel.get("test_r2"),
            }
        )
    write_csv(args.output_dir / "selection_summary.csv", selection_rows)

    print("[INFO] Selected representatives:")
    for row in selection_rows:
        print(
            f"  {row['selection_key']}: {row['run_name']} "
            f"(score={row['selection_score']:.6f})"
        )

    for group_name in GROUPS:
        items = build_group_items(group_name, selections)
        render_group(group_name, items, args.output_dir, args, plt)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
