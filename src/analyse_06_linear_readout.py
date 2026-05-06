"""
 Linear readout from latent space.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def _find(run_dir: Path, name: str):
    for candidate in [run_dir / "artifacts" / name, run_dir / name]:
        if candidate.exists():
            return candidate
    return None


def load_split(run_dir: Path, subdir: str = ""):
    """
    Load latents + predictions for one evaluation split.
    """
    base = run_dir / "artifacts"
    if subdir:
        base = base / subdir

    latent_path = base / "unmasked_latent_mean_pooled.npy"
    preds_path  = base / "eval_predictions.npz"

    if not latent_path.exists():
        raise FileNotFoundError(
            f"Missing: {latent_path}\n"
            "Re-run training with --extra-eval-splits train to generate train-split artifacts.")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing: {preds_path}")

    latents = np.load(latent_path).astype(np.float64)   # [W, 512]
    preds   = np.load(preds_path)
    targets = preds["targets"].astype(np.float64)        # [W, T, N]
    pred_rates = preds["pred_rates"].astype(np.float64)  # [W, T, N]
    heldout    = preds["heldout_neuron_indices"]          # [K]
    return latents, targets, pred_rates, heldout




def standardise(X_train: np.ndarray, X_test: np.ndarray):
    """
    Zero-mean, unit-std normalisation using train statistics only.
    """
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X_train - mu) / std, (X_test - mu) / std


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: w = (X^T X + alpha I)^{-1} X^T y  -->  [D]"""
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])



def model_window_r2(targets: np.ndarray, pred_rates: np.ndarray, n: int) -> float:
    true_win = targets[:, :, n].mean(axis=1)
    pred_win = pred_rates[:, :, n].mean(axis=1)
    return r2(true_win, pred_win)



def run_linear_readout(
    latents_train: np.ndarray,   # [W_train, D]
    targets_train: np.ndarray,   # [W_train, T, N]
    latents_test:  np.ndarray,   # [W_test, D]
    targets_test:  np.ndarray,   # [W_test, T, N]
    pred_rates_test: np.ndarray, # [W_test, T, N]
    heldout_indices: np.ndarray, # [K]
    alpha: float,
) -> list:
    """
    For every held-out neuron fit three decoders on train, evaluate on test.
    Returns a list of per-neuron dicts.
    """
    # Standardise latent features using train statistics
    X_tr_std, X_te_std = standardise(latents_train, latents_test)

    # Population mean rate per window (1-D baseline)
    pop_tr = targets_train.mean(axis=(1, 2), keepdims=True)  # [W_train, 1, 1]
    pop_te = targets_test.mean(axis=(1, 2), keepdims=True)
    pop_tr_1d = pop_tr[:, 0, 0]   # [W_train]
    pop_te_1d = pop_te[:, 0, 0]   # [W_test]
    pop_tr_feat = pop_tr_1d[:, None]   # [W_train, 1]
    pop_te_feat = pop_te_1d[:, None]   # [W_test,  1]

    rows = []
    for n in heldout_indices:
        n = int(n)
        y_tr = targets_train[:, :, n].mean(axis=1)   # [W_train] window-mean rate
        y_te = targets_test[:, :, n].mean(axis=1)    # [W_test]  window-mean rate

        mean_rate_train = float(y_tr.mean())
        mean_rate_test  = float(y_te.mean())

        # --- Model decoder (full StitchDecoder output) ---
        m_r2 = model_window_r2(targets_test, pred_rates_test, n)
        m_corr = pearson_r(y_te, pred_rates_test[:, :, n].mean(axis=1))

        # --- Latent Ridge decoder ---
        w_lat = ridge_fit(X_tr_std, y_tr, alpha)
        y_hat_lat = X_te_std @ w_lat
        lat_r2   = r2(y_te, y_hat_lat)
        lat_corr = pearson_r(y_te, y_hat_lat)

        # --- Population-rate baseline (1-D Ridge) ---
        w_pop = ridge_fit(pop_tr_feat, y_tr, alpha)
        y_hat_pop = pop_te_feat @ w_pop
        pop_r2   = r2(y_te, y_hat_pop)
        pop_corr = pearson_r(y_te, y_hat_pop)

        rows.append({
            "neuron_idx":       n,
            "mean_rate_train":  mean_rate_train,
            "mean_rate_test":   mean_rate_test,
            "model_r2":         m_r2,
            "model_corr":       m_corr,
            "latent_r2":        lat_r2,
            "latent_corr":      lat_corr,
            "poprate_r2":       pop_r2,
            "poprate_corr":     pop_corr,
            # Gain = latent over pop-rate (does latent add info beyond global activity?)
            "latent_gain_over_poprate": (
                lat_r2 - pop_r2 if not (np.isnan(lat_r2) or np.isnan(pop_r2))
                else float("nan")
            ),
            # Gap = model minus latent (how much does the decoder head add?)
            "model_minus_latent_r2": (
                m_r2 - lat_r2 if not (np.isnan(m_r2) or np.isnan(lat_r2))
                else float("nan")
            ),
        })

    return rows



def print_summary(rows, label, alpha):
    def _stats(key):
        vals = np.array([r[key] for r in rows if not np.isnan(r[key])])
        if vals.size == 0:
            return "  (no valid values)"
        return (f"  median={np.median(vals):+.4f}  mean={np.mean(vals):+.4f}"
                f"  frac>0={np.mean(vals > 0)*100:.1f}%  n={vals.size}")

    print(f"\n=== Linear Readout: {label}  (alpha={alpha}) ===")
    print(f"Model R^2   (full decoder, test windows)  :{_stats('model_r2')}")
    print(f"Latent R^2  (Ridge from 512-dim latent)   :{_stats('latent_r2')}")
    print(f"PopRate R^2 (Ridge from 1D pop mean)      :{_stats('poprate_r2')}")
    print(f"Latent gain over pop-rate                :{_stats('latent_gain_over_poprate')}")
    print(f"Model R^2 - Latent R^2 (decoder-head gap)  :{_stats('model_minus_latent_r2')}")

def write_csv(rows, output_dir):
    path = output_dir / "linear_readout_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scatter(rows, output_dir, label, plt):
    """Latent R^2 vs model R^2 — one dot per held-out neuron."""
    model_r2  = np.array([r["model_r2"]  for r in rows])
    latent_r2 = np.array([r["latent_r2"] for r in rows])
    pop_r2    = np.array([r["poprate_r2"] for r in rows])
    valid = ~(np.isnan(model_r2) | np.isnan(latent_r2))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    # Left: latent vs model
    ax = axes[0]
    ax.scatter(model_r2[valid], latent_r2[valid], s=18, alpha=0.65)
    lim = [min(np.nanmin(model_r2), np.nanmin(latent_r2)) - 0.05,
           max(np.nanmax(model_r2), np.nanmax(latent_r2)) + 0.05]
    ax.plot(lim, lim, color="black", linewidth=1.0, linestyle="--", label="y = x")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Model R^2  (full decoder)")
    ax.set_ylabel("Latent R^2  (Ridge from 512-dim latent)")
    ax.set_title(f"Linear readout vs full model\n{label}")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    # Right: population-rate vs latent
    valid2 = ~(np.isnan(pop_r2) | np.isnan(latent_r2))
    ax2 = axes[1]
    ax2.scatter(pop_r2[valid2], latent_r2[valid2], s=18, alpha=0.65, color="darkorange")
    lim2 = [min(np.nanmin(pop_r2), np.nanmin(latent_r2)) - 0.05,
            max(np.nanmax(pop_r2), np.nanmax(latent_r2)) + 0.05]
    ax2.plot(lim2, lim2, color="black", linewidth=1.0, linestyle="--", label="y = x")
    ax2.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax2.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax2.set_xlabel("Pop-rate R^2  (1-D global activity)")
    ax2.set_ylabel("Latent R^2  (Ridge from 512-dim latent)")
    ax2.set_title(f"Latent vs population-rate baseline\n{label}")
    ax2.legend(frameon=False, fontsize=8)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(output_dir / "linear_readout_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r2_histogram(rows, output_dir, label, plt):
    """Overlapping R^2 distributions for model, latent, and pop-rate decoders."""
    model_r2  = np.array([r["model_r2"]  for r in rows if not np.isnan(r["model_r2"])])
    latent_r2 = np.array([r["latent_r2"] for r in rows if not np.isnan(r["latent_r2"])])
    pop_r2    = np.array([r["poprate_r2"] for r in rows if not np.isnan(r["poprate_r2"])])

    bins = np.linspace(
        min(np.nanmin(model_r2), np.nanmin(latent_r2), np.nanmin(pop_r2)) - 0.05,
        max(np.nanmax(model_r2), np.nanmax(latent_r2), np.nanmax(pop_r2)) + 0.05,
        35,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(model_r2,  bins=bins, alpha=0.55, label=f"model  (med={np.median(model_r2):.3f})")
    ax.hist(latent_r2, bins=bins, alpha=0.55, label=f"latent (med={np.median(latent_r2):.3f})")
    ax.hist(pop_r2,    bins=bins, alpha=0.55, label=f"pop-rate (med={np.median(pop_r2):.3f})")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="R^2=0")
    ax.set_xlabel("Window-level R^2 (test set)")
    ax.set_ylabel("Neuron count")
    ax.set_title(f"R^2 distribution: three decoders\n{label}")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "linear_readout_r2_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain_by_rate(rows, output_dir, label, plt):
    """
    Latent gain over pop-rate baseline as a function of neuron firing rate.
    Shows whether the latent space adds neuron-specific information beyond
    global population activity, especially for more active neurons.
    """
    rates = np.array([r["mean_rate_train"]          for r in rows])
    gain  = np.array([r["latent_gain_over_poprate"] for r in rows])
    valid = ~np.isnan(gain)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(rates[valid], gain[valid], s=16, alpha=0.65,
                    c=gain[valid], cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--",
               label="latent = pop-rate")
    ax.set_xlabel("Mean firing rate (spk/bin, train set)")
    ax.set_ylabel("Latent R^2 − Pop-rate R^2")
    ax.set_title(f"Latent gain over global activity baseline\n{label}")
    ax.legend(frameon=False, fontsize=8)
    plt.colorbar(sc, ax=ax, label="gain", shrink=0.85)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "linear_readout_gain_by_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",        required=True, type=Path)
    ap.add_argument("--label",      default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/section6_readout"))
    ap.add_argument("--alpha",      type=float, default=1.0,
                    help="Ridge regularisation strength (default: 1.0)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = args.label or args.run.name

    # --- Load test split (primary eval) ---
    print(f"Loading test-split artifacts from {args.run} ...")
    lat_te, tgt_te, pred_te, heldout = load_split(args.run, subdir="")
    print(f"  test  latents: {lat_te.shape}   targets: {tgt_te.shape}")

    # --- Load train split ---
    print("Loading train-split artifacts ...")
    try:
        lat_tr, tgt_tr, _, heldout_tr = load_split(args.run, subdir="eval_train")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"\n[ERROR] {exc}\n\n"
            "To generate train-split latents, re-run your training script with:\n"
            "  --extra-eval-splits train\n"
            "and re-submit the job."
        ) from exc
    print(f"  train latents: {lat_tr.shape}   targets: {tgt_tr.shape}")

    # Sanity: both splits should have the same held-out neurons (same artifact seed)
    if not np.array_equal(np.sort(heldout), np.sort(heldout_tr)):
        print("[WARN] held-out neuron indices differ between train and test eval — "
              "using test-split indices for both")

    print(f"\nHeld-out neurons: {heldout.size}   Ridge alpha: {args.alpha}")

    # --- Run linear readout ---
    rows = run_linear_readout(
        latents_train=lat_tr,
        targets_train=tgt_tr,
        latents_test=lat_te,
        targets_test=tgt_te,
        pred_rates_test=pred_te,
        heldout_indices=heldout,
        alpha=args.alpha,
    )

    # --- Print + save ---
    print_summary(rows, label, args.alpha)
    write_csv(rows, args.output_dir)

    plot_scatter(rows, args.output_dir, label, plt)
    plot_r2_histogram(rows, args.output_dir, label, plt)
    plot_gain_by_rate(rows, args.output_dir, label, plt)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
