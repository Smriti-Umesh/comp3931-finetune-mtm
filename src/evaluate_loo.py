"""
Leave-One-Out (LOO) co-smoothing evaluation.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader


_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from models.ndt1 import NDT1
from train_single_session_local import LocalSessionDataset, collate_local_batch
from utils.config_utils import DictConfig


PLOT_COLORS = {
    "primary": "#2F6BFF",
    "accent": "#F05D5E",
    "neutral": "#3B3B3B",
    "grid": "#D9D9D9",
}




def rebuild_model(run_dir: Path, N: int, device: torch.device):
    meta_path = run_dir / "artifacts" / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    config = DictConfig(meta["config"])
    use_lograte = bool(config.method.model_kwargs.use_lograte)

    model = NDT1(
        config.model,
        **config.method.model_kwargs,
        num_neurons=[N],
    )
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[INFO] loaded epoch {ckpt.get('epoch','?')}  "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.6f}")
    return model, use_lograte, config


def write_eid_file():
    eid_file = _SRC.parent / "data" / "target_eids.txt"
    eid_file.parent.mkdir(parents=True, exist_ok=True)
    eid_file.write_text("control_session\n")


def _safe_label_path(label: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in label)


def _build_loo_group(
    spikes: torch.Tensor,
    n_group: List[int],
    validate_protocol: bool = False,
):
    """
    Builds the exact leave-one-neuron-out inputs for a neuron group.
    """
    B, T_bins, N_neurons = spikes.shape

    masked_list = []
    emask_list = []
    target_neurons = []
    for n in n_group:
        eval_mask = torch.zeros(B, T_bins, N_neurons, dtype=torch.int64)
        eval_mask[:, :, n] = 1
        masked = spikes.clone()
        masked[eval_mask.bool()] = 0
        masked_list.append(masked)
        emask_list.append(eval_mask)
        target_neurons.extend([n] * B)

    stacked_spikes = torch.cat(masked_list, dim=0)
    stacked_emask = torch.cat(emask_list, dim=0)

    if validate_protocol:
        repeated_spikes = spikes.repeat(len(n_group), 1, 1)
        mask_bool = stacked_emask.bool()
        target_neurons_t = torch.tensor(target_neurons, dtype=torch.int64)

        masked_bin_counts = stacked_emask.sum(dim=(1, 2))
        if not torch.all(masked_bin_counts == T_bins):
            raise RuntimeError(
                "LOO protocol violation: each stacked example must mask exactly "
                "one neuron across all time bins."
            )

        masked_neuron_counts = stacked_emask.any(dim=1).sum(dim=1)
        if not torch.all(masked_neuron_counts == 1):
            raise RuntimeError(
                "LOO protocol violation: stacked examples are masking more than "
                "one neuron."
            )

        masked_neuron_idx = stacked_emask.any(dim=1).to(torch.int64).argmax(dim=1)
        if not torch.equal(masked_neuron_idx.cpu(), target_neurons_t):
            raise RuntimeError(
                "LOO protocol violation: eval_mask does not target the expected neuron."
            )

        masked_values = stacked_spikes.masked_select(mask_bool)
        if not torch.equal(masked_values, torch.zeros_like(masked_values)):
            raise RuntimeError(
                "LOO protocol violation: masked neuron bins are not zeroed out."
            )

        if not torch.equal(
            stacked_spikes.masked_select(~mask_bool),
            repeated_spikes.masked_select(~mask_bool),
        ):
            raise RuntimeError(
                "LOO protocol violation: unmasked neuron bins changed unexpectedly."
            )

    return stacked_spikes, stacked_emask




def run_loo(model, test_loader, device, N, use_lograte, n_jobs):
    """
    For each neuron n, collect full [W, T] bin-level true counts and predicted
    rates across all test windows.

    Returns:
        true_bins  : dict {n: np.ndarray [W, T]}  true spike counts
        pred_bins  : dict {n: np.ndarray [W, T]}  model predicted rates
    """
    old_mask = getattr(model.encoder, "mask", None)
    if old_mask is not None:
        model.encoder.mask = False

    true_bins = {n: [] for n in range(N)}
    pred_bins = {n: [] for n in range(N)}

    total_groups = math.ceil(N / n_jobs)
    t0 = time.perf_counter()

    try:
        for g_idx, n_start in enumerate(range(0, N, n_jobs)):
            n_group = list(range(n_start, min(n_start + n_jobs, N)))
            k = len(n_group)

            for batch_idx, batch in enumerate(test_loader):
                spikes = batch["spikes_data"]        # [B, T, N_neurons]
                B, T_bins, N_neurons = spikes.shape

                eids = batch["eid"]
                if len(set(eids)) != 1:
                    raise RuntimeError(
                        "LOO evaluation expects one session/eid per batch so the "
                        "stacked copies match the same session metadata."
                    )

                stacked_spikes, stacked_emask = _build_loo_group(
                    spikes,
                    n_group,
                    validate_protocol=(g_idx == 0 and batch_idx == 0),
                )
                stacked_spikes = stacked_spikes.to(device)  # [B*k, T, N]
                stacked_emask = stacked_emask.to(device)

                def rep(t):
                    return t.repeat(k, *([1] * (t.dim() - 1))).to(device)

                with torch.no_grad():
                    outputs = model(
                        stacked_spikes,
                        time_attn_mask    = rep(batch["time_attn_mask"]),
                        space_attn_mask   = rep(batch["space_attn_mask"]),
                        spikes_timestamps = rep(batch["spikes_timestamps"]),
                        spikes_spacestamps= rep(batch["spikes_spacestamps"]),
                        targets           = rep(batch["target"]),
                        neuron_regions    = np.tile(batch["neuron_regions"], (k, 1)),
                        eval_mask         = stacked_emask,
                        masking_mode      = "neuron",
                        spike_augmentation= False,
                        num_neuron        = N_neurons,
                        eid               = eids[0],
                    )

                preds_raw = outputs.preds  # [B*k, T, N]
                if use_lograte:
                    pred_rates = torch.exp(torch.clamp(preds_raw, -20.0, 20.0))
                else:
                    pred_rates = torch.clamp(preds_raw, min=1e-9)
                pred_np = pred_rates.detach().cpu().numpy()
                gt_np   = spikes.numpy()   # [B, T, N]

                for i, n in enumerate(n_group):
                    # pred_np rows i*B..(i+1)*B correspond to neuron n's masked pass
                    pred_bins[n].append(pred_np[i * B:(i + 1) * B, :, n])  # [B, T]
                    true_bins[n].append(gt_np[:, :, n])                     # [B, T]

            elapsed   = time.perf_counter() - t0
            remaining = elapsed / (g_idx + 1) * (total_groups - g_idx - 1)
            print(f"  group {g_idx+1}/{total_groups}  "
                  f"neurons {n_start}–{n_start+k-1}  "
                  f"elapsed {elapsed:.0f}s  eta {remaining:.0f}s", end="\r")
    finally:
        if old_mask is not None:
            model.encoder.mask = old_mask

    print()

    # Concatenate batches → [W, T] per neuron
    return (
        {n: np.concatenate(true_bins[n], axis=0) for n in range(N)},
        {n: np.concatenate(pred_bins[n], axis=0) for n in range(N)},
    )



def _poisson_nll_sum(rate: np.ndarray, count: np.ndarray, eps: float = 1e-9) -> float:
    """
    Sum of Poisson NLL over all elements (gammaln term omitted — it cancels in bps).
    rate and count must have the same shape.
    """
    r = np.maximum(rate, eps)
    return float(np.sum(r - count * np.log(r)))


def _bps(pred_bins: np.ndarray, true_bins: np.ndarray) -> float:
    """
    IBL-MtM bits-per-spike formula.

    pred_bins, true_bins : [W, T]  (one neuron)

    null_rate = mean of true_bins over ALL windows × bins (test-set mean).
    bps = (nll_null - nll_model) / (log(2) * total_spikes)
    """
    total_spikes = float(np.nansum(true_bins))
    if total_spikes < 1e-9:
        return float("nan")

    null_rate = float(np.nanmean(true_bins))

    nll_model = _poisson_nll_sum(pred_bins, true_bins)
    nll_null  = _poisson_nll_sum(np.full_like(true_bins, null_rate), true_bins)

    return (nll_null - nll_model) / (math.log(2) * total_spikes)


def _r2(true_flat: np.ndarray, pred_flat: np.ndarray) -> float:
    """Time-resolved R^2 over all windows x time bins """
    ss_tot = float(np.sum((true_flat - true_flat.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((true_flat - pred_flat) ** 2)) / ss_tot


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(true_bins_dict: dict, pred_bins_dict: dict) -> list:
    rows = []
    for n in sorted(true_bins_dict.keys()):
        tb = true_bins_dict[n]   # [W, T]
        pb = pred_bins_dict[n]   # [W, T]

        # Window means are kept for the Pearson correlation .
        true_flat = tb.reshape(-1)     # [W*T]
        pred_flat = pb.reshape(-1)     # [W*T]
        true_win  = tb.mean(axis=1)    # [W] 
        pred_win  = pb.mean(axis=1)    # [W]

        rows.append({
            "neuron_idx":        n,
            "mean_rate":         float(tb.mean()),
            "total_spikes":      float(tb.sum()),
            "r2":                _r2(true_flat, pred_flat),
            "window_corr":       _pearson(true_win, pred_win),
            "bps_vs_null":       _bps(pb, tb),       # IBL formula, test-set null
        })
    return rows




def print_summary(rows, label):
    r2s  = np.array([r["r2"]  for r in rows if not math.isnan(r["r2"])])
    bpsv = np.array([r["bps_vs_null"] for r in rows if not math.isnan(r["bps_vs_null"])])

    print(f"\n LOO Co-smoothing: {label} ===")
    print(f"  neurons evaluated  : {len(rows)}")
    if r2s.size:
        print(f"  R^2 (all bins)      : median={np.median(r2s):+.4f}  "
              f"mean={np.mean(r2s):+.4f}  frac>0={np.mean(r2s>0)*100:.1f}%")
    if bpsv.size:
        print(f"  bits/spike (LOO)   : median={np.median(bpsv):+.4f}  "
              f"mean={np.mean(bpsv):+.4f}  frac>0={np.mean(bpsv>0)*100:.1f}%")


def write_csv(rows, output_dir):
    path = output_dir / "loo_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows



def _style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=PLOT_COLORS["grid"], alpha=0.5, linewidth=0.8)


def plot_r2_histogram(rows, output_dir, plt):
    r2s = np.array([r["r2"] for r in rows if not math.isnan(r["r2"])])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(r2s, bins=35, color=PLOT_COLORS["primary"], alpha=0.82,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color=PLOT_COLORS["neutral"], linewidth=1.2, linestyle="--", label="R² = 0")
    ax.axvline(float(np.median(r2s)), color=PLOT_COLORS["accent"], linewidth=1.5,
               label=f"median={np.median(r2s):.3f}")
    ax.set_xlabel("LOO R²")
    ax.set_ylabel("Neuron count")
    ax.set_title("LOO R² Distribution")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "loo_r2_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r2_vs_rate(rows, output_dir, plt):
    r2s   = np.array([r["r2"] for r in rows])
    rates = np.array([r["mean_rate"]  for r in rows])
    valid = ~np.isnan(r2s)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sc = ax.scatter(
        rates[valid],
        r2s[valid],
        s=20,
        alpha=0.72,
        c=r2s[valid],
        cmap="viridis",
        vmin=min(-0.1, float(np.nanmin(r2s[valid]))) if valid.any() else -0.1,
        vmax=max(0.2, float(np.nanmax(r2s[valid]))) if valid.any() else 0.2,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.axhline(0.0, color=PLOT_COLORS["neutral"], linewidth=1.0, linestyle="--")
    ax.set_xlabel("Mean firing rate (spk/bin, test windows)")
    ax.set_ylabel("LOO R^2")
    ax.set_title("LOO R^2 vs Mean Firing Rate")
    plt.colorbar(sc, ax=ax, label="R^2")
    _style(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(output_dir / "loo_r2_vs_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bps_histogram(rows, output_dir, plt):
    bps = np.array([r["bps_vs_null"] for r in rows
                    if not math.isnan(r["bps_vs_null"])])
    if bps.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(bps, bins=35, color="#2AA889", alpha=0.84,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color=PLOT_COLORS["neutral"], linewidth=1.2, linestyle="--",
               label="0 bps")
    ax.axvline(float(np.median(bps)), color=PLOT_COLORS["accent"], linewidth=1.5,
               label=f"median={np.median(bps):.4f}")
    ax.set_xlabel("LOO bits / spike")
    ax.set_ylabel("Neuron count")
    ax.set_title("LOO Bits per Spike Distribution")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "loo_bps_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",         required=True, type=Path)
    ap.add_argument("--data-dir",    required=True, type=str)
    ap.add_argument("--label",       default=None)
    ap.add_argument("--output-dir",  type=Path, default=Path("results_20ms/report/loo"))
    ap.add_argument("--batch-size",  type=int, default=8)
    ap.add_argument("--n-jobs",      type=int, default=16,
                    help="Neurons per forward pass (higher = faster, more GPU memory)")
    ap.add_argument("--split",       type=str, default="test",
                    choices=["test", "val"])
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    label = args.label or args.run.name
    run_output_dir = args.output_dir / _safe_label_path(label)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = run_output_dir / "mpl_cache"
    mpl_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    write_eid_file()

    print(f"[INFO] loading {args.split} split from {args.data_dir} ...")
    ds = LocalSessionDataset(args.data_dir, split=args.split)
    N  = ds.N
    print(f"[INFO] {args.split} windows: {len(ds)}   neurons: {N}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=collate_local_batch)

    print(f"[INFO] rebuilding model ...")
    model, use_lograte, _config = rebuild_model(args.run, N, device)

    print(f"\n[INFO] LOO evaluation: {N} neurons, n_jobs={args.n_jobs}, "
          f"~{math.ceil(N/args.n_jobs)} groups × {math.ceil(len(ds)/args.batch_size)} batches")

    true_bins, pred_bins = run_loo(model, loader, device, N, use_lograte, args.n_jobs)

    rows = compute_metrics(true_bins, pred_bins)
    print_summary(rows, label)
    write_csv(rows, run_output_dir)

    plot_r2_histogram(rows, run_output_dir, plt)
    plot_r2_vs_rate(rows, run_output_dir, plt)
    plot_bps_histogram(rows, run_output_dir, plt)

    print(f"\n[DONE] saved to: {run_output_dir}")


if __name__ == "__main__":
    main()
