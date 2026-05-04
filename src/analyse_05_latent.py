"""
Section 5: Latent space analysis.

Loads the unmasked latent representations saved by local_eval_artifacts.py
and asks three questions:

  1. How much variance is captured by the leading PCs?  (scree plot)
  2. Does the latent space track time / population activity?
     (PC1–PC2 scatter coloured by window index and mean firing rate)
  3. Which held-out neurons are most linearly encoded in the top latent dims?
     (per-neuron correlation with PC projections)
  4. What does the within-window temporal trajectory look like?
     (PCA of the full [T=100, 512] latent sequence for one exemplar window)

Inputs (from a single run directory):
  - artifacts/unmasked_latent_mean_pooled.npy   [W, 512]
  - artifacts/unmasked_latent_sequence.npz      {latents: [W, T, 512]}  (for trajectory)
  - artifacts/eval_predictions.npz              {targets, pred_rates, eval_mask,
                                                  heldout_neuron_indices, baseline_per_neuron}

Outputs (all written to --output-dir):
  - latent_scree.png            : % variance explained per PC
  - latent_pca_time.png         : PC1 vs PC2 coloured by window index (chronological)
  - latent_pca_rate.png         : PC1 vs PC2 coloured by mean population firing rate
  - latent_tsne_time.png        : t-SNE embedding coloured by window index
  - latent_tsne_rate.png        : t-SNE embedding coloured by mean population firing rate
  - latent_umap_time.png        : UMAP embedding coloured by window index
  - latent_umap_rate.png        : UMAP embedding coloured by mean population firing rate
  - latent_neuron_alignment.png : top held-out neurons ranked by |corr| with PC1
  - latent_trajectory.png       : temporal trajectory within one exemplar window
  - latent_alignment.csv        : per-neuron correlation with top-5 PCs

Usage:
    python src/analyse_05_latent.py \\
        --run   results/local_neuron_mask_full_plain_lr5e-5_4326037 \\
        --label "MtM-neuron" \\
        --output-dir results/report/section5_latent
"""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _find(run_dir: Path, name: str):
    # Flat layout:   artifacts/<name>           (older single-mask runs)
    # Nested layout: artifacts/eval_test_neuron/<name>  (combined-mask runs)
    # For combined runs we prefer the neuron-masking eval because it gives
    # heldout_neuron_indices for the alignment plot.
    for candidate in [
        run_dir / "artifacts" / name,
        run_dir / "artifacts" / "eval_test_neuron" / name,
        run_dir / name,
    ]:
        if candidate.exists():
            return candidate
    return None


def load_latents(run_dir: Path):
    path = _find(run_dir, "unmasked_latent_mean_pooled.npy")
    if path is None:
        raise FileNotFoundError(
            f"unmasked_latent_mean_pooled.npy not found under {run_dir}.\n"
            "Re-run evaluation with local_eval_artifacts.py to generate it.")
    return np.load(path).astype(np.float64)  # [W, 512]


def load_latent_sequence(run_dir: Path):
    path = _find(run_dir, "unmasked_latent_sequence.npz")
    if path is None:
        return None
    data = np.load(path)
    key = "latents" if "latents" in data else list(data.keys())[0]
    return data[key].astype(np.float64)  # [W, T, 512]


def load_predictions(run_dir: Path):
    path = _find(run_dir, "eval_predictions.npz")
    if path is None:
        raise FileNotFoundError(f"eval_predictions.npz not found under {run_dir}")
    return np.load(path)


# ---------------------------------------------------------------------------
# PCA via full SVD
# ---------------------------------------------------------------------------

def run_pca(X):
    """
    Centre X [N, D] and return (coords, pct_var, components, mean).

    coords      : [N, K]  projections onto all K = min(N, D) PCs
    pct_var     : [K]     % variance explained per PC
    components  : [K, D]  unit-norm PC directions (rows)
    mean        : [D]     column means used for centring
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    pct_var = 100.0 * var / var.sum()
    coords = U * s          # equivalent to Xc @ Vt.T
    return coords, pct_var, Vt, mean


def run_tsne(X, random_state=0, perplexity=30.0):
    """
    Run 2D t-SNE on latent vectors.

    We first project to a modest PCA subspace for stability/speed, then fit t-SNE.
    """
    n_samples, n_features = X.shape
    if n_samples < 4:
        raise ValueError("Need at least 4 windows for t-SNE")

    pca_dim = min(50, n_features, max(2, n_samples - 1))
    pca_coords, _, _, _ = run_pca(X)
    X_pre = pca_coords[:, :pca_dim]

    max_valid_perplexity = max(2.0, min(float(perplexity), float(n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=max_valid_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X_pre), max_valid_perplexity


def run_umap(X, random_state=0, n_neighbors=15, min_dist=0.1):
    """
    Run 2D UMAP on latent vectors if umap-learn is installed.
    """
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", Path(tempfile.gettempdir()) / "numba_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))

    try:
        import umap
    except ModuleNotFoundError:
        return None, "umap-learn is not installed"

    n_samples = X.shape[0]
    if n_samples < 4:
        raise ValueError("Need at least 4 windows for UMAP")

    n_neighbors = max(2, min(int(n_neighbors), n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X), None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scree(pct_var, output_dir, label, plt, n_show=30):
    k = min(n_show, len(pct_var))
    cumvar = np.cumsum(pct_var)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(np.arange(1, k + 1), pct_var[:k])
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("% variance explained")
    axes[0].set_title(f"Scree plot (top {k} PCs)\n{label}")
    _style(axes[0])

    axes[1].plot(np.arange(1, k + 1), cumvar[:k], marker="o", markersize=3)
    axes[1].axhline(90.0, color="red", linewidth=0.8, linestyle="--", label="90%")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative % variance")
    axes[1].set_title("Cumulative variance")
    axes[1].legend(frameon=False, fontsize=8)
    _style(axes[1])

    fig.tight_layout()
    fig.savefig(output_dir / "latent_scree.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    for threshold in [50, 75, 90]:
        n_needed = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"  PCs needed for {threshold}% variance: {n_needed}")


def _scatter_colored(ax, x, y, c, cmap, label_cb, title, plt):
    sc = ax.scatter(x, y, c=c, cmap=cmap, s=10, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=label_cb, shrink=0.85)
    _style(ax)


def plot_pca_scatter(coords, targets, output_dir, label, plt):
    W = coords.shape[0]
    pc1, pc2 = coords[:, 0], coords[:, 1]

    # Colour 1: chronological window index (proxy for recording time)
    window_idx = np.arange(W, dtype=float)

    # Colour 2: mean population firing rate per window
    pop_rate = targets.mean(axis=(1, 2))  # [W]

    fig_time, ax_time = plt.subplots(figsize=(6.0, 5.0))
    _scatter_colored(ax_time, pc1, pc2, window_idx, "viridis",
                     "window index (time)", f"Latent PC1–PC2 coloured by time\n{label}", plt)
    fig_time.tight_layout()
    fig_time.savefig(output_dir / "latent_pca_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig_time)

    fig_rate, ax_rate = plt.subplots(figsize=(6.0, 5.0))
    _scatter_colored(ax_rate, pc1, pc2, pop_rate, "plasma",
                     "mean pop. rate (spk/bin)", f"Latent PC1–PC2 coloured by firing rate\n{label}", plt)
    fig_rate.tight_layout()
    fig_rate.savefig(output_dir / "latent_pca_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rate)

    # Correlation with time and rate
    for name, col in [("window_idx", window_idx), ("pop_rate", pop_rate)]:
        for i, vec in enumerate([pc1, pc2]):
            c = float(np.corrcoef(vec, col)[0, 1])
            print(f"  PC{i+1} vs {name}: r={c:+.3f}")


def plot_embedding_scatter(coords, targets, output_dir, label, plt, stem, title_prefix):
    window_idx = np.arange(coords.shape[0], dtype=float)
    pop_rate = targets.mean(axis=(1, 2))

    fig_time, ax_time = plt.subplots(figsize=(6.0, 5.0))
    _scatter_colored(ax_time, coords[:, 0], coords[:, 1], window_idx, "viridis",
                     "window index (time)", f"{title_prefix} coloured by time\n{label}", plt)
    ax_time.set_xlabel(f"{stem.upper()}-1")
    ax_time.set_ylabel(f"{stem.upper()}-2")
    fig_time.tight_layout()
    fig_time.savefig(output_dir / f"latent_{stem}_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig_time)

    fig_rate, ax_rate = plt.subplots(figsize=(6.0, 5.0))
    _scatter_colored(ax_rate, coords[:, 0], coords[:, 1], pop_rate, "plasma",
                     "mean pop. rate (spk/bin)", f"{title_prefix} coloured by firing rate\n{label}", plt)
    ax_rate.set_xlabel(f"{stem.upper()}-1")
    ax_rate.set_ylabel(f"{stem.upper()}-2")
    fig_rate.tight_layout()
    fig_rate.savefig(output_dir / f"latent_{stem}_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig_rate)


def plot_neuron_alignment(coords, targets, heldout_indices, output_dir, label, plt,
                          n_show=20):
    """
    For each held-out neuron compute the Pearson correlation between its
    window-mean firing rate and each of PC1–PC5, then rank neurons by |corr| with PC1.
    """
    n_pcs = min(5, coords.shape[1])
    rows = []
    for n in heldout_indices:
        true_win = targets[:, :, n].mean(axis=1)  # [W]
        if true_win.std() < 1e-9:
            continue
        corrs = []
        for p in range(n_pcs):
            pc = coords[:, p]
            if pc.std() < 1e-9:
                corrs.append(float("nan"))
            else:
                corrs.append(float(np.corrcoef(true_win, pc)[0, 1]))
        rows.append({"neuron_idx": int(n),
                     "mean_rate": float(true_win.mean()),
                     **{f"corr_pc{p+1}": corrs[p] for p in range(n_pcs)}})

    if not rows:
        print("  [WARN] No valid neurons for alignment plot")
        return rows

    # Sort by |corr_pc1|
    rows.sort(key=lambda r: abs(r.get("corr_pc1", 0.0)), reverse=True)
    top = rows[:n_show]

    neuron_labels = [str(r["neuron_idx"]) for r in top]
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(top)), 4.6))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_pcs))
    for p in range(n_pcs):
        vals = [r.get(f"corr_pc{p+1}", float("nan")) for r in top]
        ax.bar(x + (p - n_pcs / 2 + 0.5) * width, vals, width=width,
               label=f"PC{p+1}", color=colors[p])

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(neuron_labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Held-out neuron index")
    ax.set_ylabel("Pearson r (window-mean rate vs PC)")
    ax.set_title(f"Neuron–latent alignment (top {len(top)} by |PC1| corr)\n{label}")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_neuron_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_trajectory(latent_seq, output_dir, label, plt, window_idx=None):
    """
    PCA on the T=100 latent states within a single window to show
    within-window temporal dynamics.

    latent_seq : [W, T, D]
    window_idx : which window to use (default: the one with highest mean norm)
    """
    W, T, D = latent_seq.shape

    if window_idx is None:
        # Pick window with largest latent norm variance (most dynamic)
        norms = np.linalg.norm(latent_seq, axis=-1)  # [W, T]
        window_idx = int(np.argmax(norms.var(axis=1)))

    seq = latent_seq[window_idx]  # [T, D]
    mean_s = seq.mean(axis=0)
    seq_c  = seq - mean_s
    U, s, Vt = np.linalg.svd(seq_c, full_matrices=False)
    coords = U * s  # [T, 2+]

    t = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Trajectory in PC1–PC2 plane
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=t, cmap="viridis", s=20, zorder=3)
    axes[0].plot(coords[:, 0], coords[:, 1], linewidth=0.6, alpha=0.4, color="grey")
    axes[0].scatter(coords[0, 0], coords[0, 1], s=60, marker="^", color="green",
                    zorder=5, label="t=0")
    axes[0].scatter(coords[-1, 0], coords[-1, 1], s=60, marker="s", color="red",
                    zorder=5, label="t=T-1")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title(f"Within-window trajectory (window {window_idx})")
    axes[0].legend(frameon=False, fontsize=8)
    plt.colorbar(sc, ax=axes[0], label="time bin", shrink=0.85)
    _style(axes[0])

    # PC1 and PC2 time series
    axes[1].plot(t, coords[:, 0], label="PC1", linewidth=1.5)
    axes[1].plot(t, coords[:, 1], label="PC2", linewidth=1.5)
    axes[1].set_xlabel("Time bin (20 ms each)")
    axes[1].set_ylabel("PC projection")
    axes[1].set_title("PC1/PC2 over time within window")
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1])

    pct = 100.0 * (s ** 2) / max((s ** 2).sum(), 1e-12)
    fig.suptitle(f"{label} — window {window_idx} "
                 f"(PC1={pct[0]:.1f}%, PC2={pct[1]:.1f}% of within-window variance)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def write_alignment_csv(rows, output_dir):
    if not rows:
        return
    path = output_dir / "latent_alignment.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",         required=True, type=Path)
    ap.add_argument("--label",       default=None)
    ap.add_argument("--output-dir",  type=Path, default=Path("results_20ms/report/section5_latent"))
    ap.add_argument("--n-neurons",   type=int, default=20,
                    help="Top N held-out neurons to show in alignment plot")
    ap.add_argument("--traj-window", type=int, default=None,
                    help="Window index for trajectory plot (default: most dynamic)")
    ap.add_argument("--tsne",        action="store_true",
                    help="Also run 2D t-SNE on the latent vectors")
    ap.add_argument("--tsne-perplexity", type=float, default=30.0,
                    help="Requested t-SNE perplexity; clipped to a valid range for the run")
    ap.add_argument("--umap",        action="store_true",
                    help="Also run 2D UMAP on the latent vectors if umap-learn is installed")
    ap.add_argument("--umap-neighbors", type=int, default=15,
                    help="UMAP n_neighbors (if --umap is enabled)")
    ap.add_argument("--umap-min-dist", type=float, default=0.1,
                    help="UMAP min_dist (if --umap is enabled)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = args.label or args.run.name

    # --- Load ---
    print(f"Loading latents from {args.run} ...")
    latents = load_latents(args.run)          # [W, 512]
    preds   = load_predictions(args.run)
    targets = preds["targets"].astype(np.float64)              # [W, T, N]
    heldout = preds["heldout_neuron_indices"]                   # [K]

    W, D = latents.shape
    print(f"  latent mean-pooled shape : {latents.shape}")
    print(f"  targets shape            : {targets.shape}")
    print(f"  held-out neurons         : {heldout.size}")

    # --- PCA ---
    print("\nRunning PCA ...")
    coords, pct_var, components, mean = run_pca(latents)
    print(f"  PC1 explains {pct_var[0]:.1f}%  |  PC2 {pct_var[1]:.1f}%")

    # --- Scree ---
    print("\nScree / cumulative variance:")
    plot_scree(pct_var, args.output_dir, label, plt)

    # --- PCA scatter ---
    print("\nPCA scatter correlations:")
    plot_pca_scatter(coords, targets, args.output_dir, label, plt)

    if args.tsne:
        print("\nRunning t-SNE ...")
        tsne_coords, used_perplexity = run_tsne(
            latents,
            random_state=0,
            perplexity=args.tsne_perplexity,
        )
        print(f"  t-SNE shape             : {tsne_coords.shape}")
        print(f"  t-SNE perplexity used   : {used_perplexity:.1f}")
        plot_embedding_scatter(
            tsne_coords,
            targets,
            args.output_dir,
            label,
            plt,
            stem="tsne",
            title_prefix="Latent t-SNE",
        )

    if args.umap:
        print("\nRunning UMAP ...")
        umap_coords, umap_error = run_umap(
            latents,
            random_state=0,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )
        if umap_error is not None:
            print(f"  [WARN] skipping UMAP: {umap_error}")
        else:
            print(f"  UMAP shape              : {umap_coords.shape}")
            plot_embedding_scatter(
                umap_coords,
                targets,
                args.output_dir,
                label,
                plt,
                stem="umap",
                title_prefix="Latent UMAP",
            )

    # --- Neuron alignment ---
    print(f"\nNeuron–latent alignment (top {args.n_neurons}):")
    alignment_rows = plot_neuron_alignment(
        coords, targets, heldout, args.output_dir, label, plt, n_show=args.n_neurons)
    if alignment_rows:
        top5 = alignment_rows[:5]
        print(f"  Top neurons by |PC1 corr|: "
              + ", ".join(f"n{r['neuron_idx']} r={r.get('corr_pc1', float('nan')):+.3f}"
                          for r in top5))
    write_alignment_csv(alignment_rows, args.output_dir)

    # --- Temporal trajectory ---
    latent_seq = load_latent_sequence(args.run)
    if latent_seq is not None:
        print(f"\nTemporal trajectory (latent_sequence shape: {latent_seq.shape}) ...")
        plot_trajectory(latent_seq, args.output_dir, label, plt, window_idx=args.traj_window)
    else:
        print("\n[INFO] unmasked_latent_sequence.npz not found — skipping trajectory plot")
        print("       Re-run evaluation to generate it.")

    preds.close()
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
