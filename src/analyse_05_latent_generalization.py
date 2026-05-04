"""
 Latent-space generalisation by split.

For one run, this script asks the key latent generalisation question:
do train/val/test windows intermix in latent space, or do they cluster apart?

It loads latent vectors for the available evaluation splits, balances the number
of windows per split, fits a shared PCA basis, then produces shared PCA / UMAP /
t-SNE plots coloured by split and by mean population firing rate.

"""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE


SPLIT_COLORS = {
    "train": "#2563eb",
    "val": "#f59e0b",
    "test": "#dc2626",
}


def resolve_split_base(run_dir: Path, split: str, mode: str | None):
    artifacts = run_dir / "artifacts"
    candidates = []
    if split == "test":
        if mode in {"neuron", "causal"}:
            candidates.append(artifacts / f"eval_test_{mode}")
        candidates.append(artifacts)
    else:
        if mode in {"neuron", "causal"}:
            candidates.append(artifacts / f"eval_{split}_{mode}")
        candidates.append(artifacts / f"eval_{split}")

    for base in candidates:
        if (base / "unmasked_latent_mean_pooled.npy").exists() and (base / "eval_predictions.npz").exists():
            return base
    return None


def load_split(run_dir: Path, split: str, mode: str | None):
    base = resolve_split_base(run_dir, split, mode)
    if base is None:
        return None
    latents = np.load(base / "unmasked_latent_mean_pooled.npy").astype(np.float64)
    with np.load(base / "eval_predictions.npz") as preds:
        targets = preds["targets"].astype(np.float64)
    mean_rates = targets.mean(axis=(1, 2))
    return latents, mean_rates


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
    return coords[:, :n_components], explained[:n_components]


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


def plot_split_panel(ax, coords, rows, title):
    split_order = [split for split in ("train", "val", "test") if any(r["split"] == split for r in rows)]
    for split in split_order:
        idx = [i for i, row in enumerate(rows) if row["split"] == split]
        pts = coords[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            alpha=0.8,
            color=SPLIT_COLORS[split],
            label=split,
        )
    ax.set_title(title)
    _style(ax)


def plot_rate_panel(ax, coords, rates, title, plt):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=rates, cmap="viridis", s=18, alpha=0.85)
    ax.set_title(title)
    _style(ax)
    plt.colorbar(sc, ax=ax, label="Mean population firing rate", shrink=0.85)


def plot_embedding(coords, rows, rates, title_prefix, out_path, label, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    plot_split_panel(axes[0], coords, rows, f"{title_prefix} coloured by split")
    plot_rate_panel(axes[1], coords, rates, f"{title_prefix} coloured by mean rate", plt)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    if label:
        fig.suptitle(label, fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 1))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--mode", choices=["neuron", "causal"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/latent_generalization"))
    parser.add_argument("--pca-dims", type=int, default=40)
    parser.add_argument("--max-points-per-split", type=int, default=None)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--l2-normalise", action="store_true")
    parser.add_argument("--hide-suptitle", action="store_true",
                        help="Do not print the run label as a large figure title")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = args.label or args.run.name
    plot_label = "" if args.hide_suptitle else label
    split_data = {}
    for split in ("train", "val", "test"):
        loaded = load_split(args.run, split, args.mode)
        if loaded is not None:
            split_data[split] = loaded

    if "train" not in split_data or "test" not in split_data:
        raise SystemExit(
            "Need at least train and test latent artifacts for split-generalization analysis."
        )

    min_count = min(latents.shape[0] for latents, _ in split_data.values())
    n_points = min_count if args.max_points_per_split is None else min(min_count, args.max_points_per_split)
    rng = np.random.default_rng(args.random_seed)

    rows = []
    X_parts = []
    rate_parts = []
    summary_rows = []
    for split, (latents, mean_rates) in split_data.items():
        if n_points < latents.shape[0]:
            indices = np.sort(rng.choice(latents.shape[0], size=n_points, replace=False))
        else:
            indices = np.arange(latents.shape[0])
        X = latents[indices]
        if args.l2_normalise:
            X = maybe_l2_normalise(X)
        rates = mean_rates[indices]
        X_parts.append(X)
        rate_parts.append(rates)
        summary_rows.append({"split": split, "n_windows_used": int(X.shape[0])})
        for local_index, (window_index, rate) in enumerate(zip(indices, rates)):
            rows.append(
                {
                    "split": split,
                    "window_index": int(window_index),
                    "mean_rate": float(rate),
                    "local_index": int(local_index),
                }
            )

    X_all = np.concatenate(X_parts, axis=0)
    rates_all = np.concatenate(rate_parts, axis=0)
    pca_coords, explained = run_pca(X_all, args.pca_dims)
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
        print(f"[WARN] skipping UMAP: {umap_error}")
        umap_coords = np.full((X_all.shape[0], 2), np.nan)
    if tsne_coords is None:
        print(f"[WARN] skipping t-SNE: {tsne_error}")
        tsne_coords = np.full((X_all.shape[0], 2), np.nan)

    for row, pca_xy, umap_xy, tsne_xy in zip(rows, pca_2d, umap_coords, tsne_coords):
        row["pca1"] = float(pca_xy[0])
        row["pca2"] = float(pca_xy[1])
        row["umap1"] = float(umap_xy[0]) if np.isfinite(umap_xy[0]) else float("nan")
        row["umap2"] = float(umap_xy[1]) if np.isfinite(umap_xy[1]) else float("nan")
        row["tsne1"] = float(tsne_xy[0]) if np.isfinite(tsne_xy[0]) else float("nan")
        row["tsne2"] = float(tsne_xy[1]) if np.isfinite(tsne_xy[1]) else float("nan")

    write_csv(args.output_dir / "split_embedding_points.csv", rows)
    write_csv(args.output_dir / "split_counts.csv", summary_rows)
    write_csv(
        args.output_dir / "pca_scree.csv",
        [{"pc": i + 1, "fraction_variance_explained": float(v)} for i, v in enumerate(explained[: min(20, len(explained))])],
    )

    title_suffix = " (L2-normalised)" if args.l2_normalise else ""
    single_plot_label = (plot_label + title_suffix) if plot_label else ""
    plot_embedding(pca_2d, rows, rates_all, "Shared PCA", args.output_dir / "latent_split_pca.png", single_plot_label, plt)
    plot_embedding(umap_coords, rows, rates_all, "Shared UMAP", args.output_dir / "latent_split_umap.png", single_plot_label, plt)
    plot_embedding(tsne_coords, rows, rates_all, "Shared t-SNE", args.output_dir / "latent_split_tsne.png", single_plot_label, plt)

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 15.8))
    panels = [
        ("Shared PCA", pca_2d),
        ("Shared UMAP", umap_coords),
        ("Shared t-SNE", tsne_coords),
    ]
    for row_idx, (name, coords) in enumerate(panels):
        plot_split_panel(axes[row_idx, 0], coords, rows, f"{name} coloured by split")
        plot_rate_panel(axes[row_idx, 1], coords, rates_all, f"{name} coloured by mean rate", plt)
        if row_idx == 0:
            axes[row_idx, 0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if plot_label:
        fig.suptitle(f"{plot_label}: latent split intermixing{title_suffix}", fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.output_dir / "latent_split_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
