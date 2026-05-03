import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ARTIFACT_DIRNAME = "artifacts"


def _artifact_dir(save_dir: str, subdir: Optional[str] = None) -> Path:
    path = Path(save_dir) / ARTIFACT_DIRNAME
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_matplotlib_cache(artifact_dir: Path) -> None:
    mpl_dir = artifact_dir / "matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_run_metadata(save_dir: str, args: Any, config: Dict[str, Any], split_lengths: Dict[str, int]) -> None:
    artifact_dir = _artifact_dir(save_dir)
    payload = {
        "args": _json_ready(vars(args)),
        "config": _json_ready(dict(config)),
        "split_lengths": _json_ready(split_lengths),
    }
    with open(artifact_dir / "run_metadata.json", "w") as f:
        json.dump(payload, f, indent=2)


def save_pretrained_load_report(save_dir: str, load_report: Dict[str, Any]) -> None:
    # Keep the transfer audit separate from stdout so later analyses can verify what actually loaded.
    artifact_dir = _artifact_dir(save_dir)
    with open(artifact_dir / "pretrained_load_report.json", "w") as f:
        json.dump(_json_ready(load_report), f, indent=2)


def save_history_artifacts(save_dir: str, history: List[Dict[str, float]]) -> None:
    # Save history every epoch 
    artifact_dir = _artifact_dir(save_dir)
    if not history:
        return

    
    base_fields = ["epoch", "train_loss_per_masked_bin", "val_loss_per_masked_bin", "best_val_loss_so_far"]
    extra_fields = []
    for row in history:
        for key in row.keys():
            if key not in base_fields and key not in extra_fields:
                extra_fields.append(key)
    fieldnames = base_fields + extra_fields
    with open(artifact_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    with open(artifact_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    _prepare_matplotlib_cache(artifact_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for loss plot: {exc}")
        return

    epochs = [row["epoch"] for row in history]
    train = [row["train_loss_per_masked_bin"] for row in history]
    val = [row["val_loss_per_masked_bin"] for row in history]
    best_idx = int(np.nanargmin(val))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(epochs, train, label="train", linewidth=2.0)
    ax.plot(epochs, val, label="val", linewidth=2.0)
    ax.scatter([epochs[best_idx]], [val[best_idx]], s=55, label=f"best val {val[best_idx]:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss per masked bin")
    ax.set_title("Training history")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(artifact_dir / "history_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_train_baselines(train_dataset: Any, chunk_size: int = 64) -> Dict[str, np.ndarray]:
    # Compute train-only rate baselines without loading the full session into memory at once.
    T, N = int(train_dataset.T), int(train_dataset.N)
    sum_global = 0.0
    count_global = 0
    sum_neuron = np.zeros(N, dtype=np.float64)
    sum_time_neuron = np.zeros((T, N), dtype=np.float64)

    for start in range(0, len(train_dataset.indices), chunk_size):
        idx = train_dataset.indices[start:start + chunk_size]
        batch = np.asarray(train_dataset.spikes_data[idx], dtype=np.float64)
        sum_global += float(batch.sum())
        count_global += int(batch.size)
        sum_neuron += batch.sum(axis=(0, 1))
        sum_time_neuron += batch.sum(axis=0)

    return {
        "global": np.asarray(sum_global / max(count_global, 1), dtype=np.float64),
        "per_neuron": sum_neuron / max(len(train_dataset.indices) * T, 1),
        "per_time_neuron": sum_time_neuron / max(len(train_dataset.indices), 1),
    }


def _make_eval_mask(
    batch_shape: Tuple[int, int, int],
    masking_mode: str,
    mask_ratio: float,
    rng: np.random.Generator,
    cached_selection: Dict[str, np.ndarray],
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    B, T, N = batch_shape
    mask = torch.zeros((B, T, N), dtype=torch.int64)

    if masking_mode == "neuron":
        if "neuron_indices" not in cached_selection:
            n_mask = max(1, int(round(mask_ratio * N)))
            cached_selection["neuron_indices"] = np.sort(rng.choice(N, size=min(n_mask, N), replace=False))
        mask[:, :, cached_selection["neuron_indices"]] = 1
    elif masking_mode == "causal":
        if "time_indices" not in cached_selection:
            n_mask = max(1, int(round(mask_ratio * T)))
            cached_selection["time_indices"] = np.arange(max(T - n_mask, 0), T, dtype=np.int64)
        mask[:, cached_selection["time_indices"], :] = 1
    else:
        raise ValueError(f"Unsupported local artifact masking mode: {masking_mode}")

    return mask, cached_selection


def _poisson_nll_from_rate(rate: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    safe_rate = np.maximum(rate, eps)
    return safe_rate - target * np.log(safe_rate)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _window_r2_for_neuron(true_win: np.ndarray, pred_win: np.ndarray) -> float:
    """Window-level R^2: matches the IBL-MtM co-smoothing metric (trial-averaged R^2    )."""
    ss_tot = float(np.sum((true_win - true_win.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(np.sum((true_win - pred_win) ** 2))
    return 1.0 - ss_res / ss_tot


def _compute_window_r2(
    targets: np.ndarray,
    pred_rates: np.ndarray,
    eval_mask: np.ndarray,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """
    Compute window-level R^2 for every neuron that has any masked bins.

    Returns:
        per_neuron : {neuron_idx: r2}
        aggregate  : {median_window_r2, mean_window_r2, frac_window_r2_positive,
                      n_neurons_evaluated}
    """
    y    = targets.astype(np.float64)
    pred = pred_rates.astype(np.float64)
    mask = eval_mask.astype(bool)  # [W, T, N]

    per_neuron: Dict[int, float] = {}
    for n in range(y.shape[2]):
        m_n = mask[:, :, n]          # [W, T] — which (window, time) pairs are masked
        if not m_n.any():
            continue
        # Mean over the masked time-steps within each window.
        # For neuron-masking this is mean over all 100 bins; for causal it is mean
        # over the held-out future bins.  Both use only masked positions.
        with np.errstate(invalid="ignore"):
            true_win = np.where(m_n.any(axis=1, keepdims=True),
                                (y[:, :, n] * m_n).sum(axis=1) / np.maximum(m_n.sum(axis=1), 1),
                                np.nan)  # [W]
            pred_win = np.where(m_n.any(axis=1, keepdims=True),
                                (pred[:, :, n] * m_n).sum(axis=1) / np.maximum(m_n.sum(axis=1), 1),
                                np.nan)
        # Only windows where the neuron actually had masked bins
        valid = m_n.any(axis=1)
        per_neuron[n] = _window_r2_for_neuron(true_win[valid], pred_win[valid])

    valid_r2 = np.array([v for v in per_neuron.values() if not np.isnan(v)])
    aggregate: Dict[str, float] = {
        "n_neurons_evaluated_r2": float(len(valid_r2)),
        "median_window_r2":       float(np.median(valid_r2))       if valid_r2.size else float("nan"),
        "mean_window_r2":         float(np.mean(valid_r2))         if valid_r2.size else float("nan"),
        "frac_window_r2_positive": float(np.mean(valid_r2 > 0))    if valid_r2.size else float("nan"),
    }
    return per_neuron, aggregate


def _summarise_metrics(
    targets: np.ndarray,
    pred_rates: np.ndarray,
    eval_mask: np.ndarray,
    baselines: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]]]:
    # Compute aggregate, per-neuron, and per-time metrics from saved arrays 
    mask_bool = eval_mask.astype(bool)
    y = targets.astype(np.float64)
    pred = pred_rates.astype(np.float64)

    global_rate = np.full_like(y, float(baselines["global"]), dtype=np.float64)
    per_neuron_rate = np.broadcast_to(baselines["per_neuron"][None, None, :], y.shape).astype(np.float64)
    per_time_neuron_rate = np.broadcast_to(baselines["per_time_neuron"][None, :, :], y.shape).astype(np.float64)

    model_nll = _poisson_nll_from_rate(pred, y)
    global_nll = _poisson_nll_from_rate(global_rate, y)
    per_neuron_nll = _poisson_nll_from_rate(per_neuron_rate, y)
    per_time_neuron_nll = _poisson_nll_from_rate(per_time_neuron_rate, y)


    test_null_rate = y.mean(axis=(0, 1), keepdims=False)   # [N]
    test_null_nll = _poisson_nll_from_rate(
        np.broadcast_to(test_null_rate[None, None, :], y.shape).astype(np.float64), y
    )

    masked_spikes = float(y[mask_bool].sum())
    model_sum = float(model_nll[mask_bool].sum())
    test_null_sum = float(test_null_nll[mask_bool].sum())

    summary = {
        "masked_bins": int(mask_bool.sum()),
        "masked_spikes": masked_spikes,
        "model_nll_per_masked_bin": float(model_nll[mask_bool].mean()),
        "global_mean_nll_per_masked_bin": float(global_nll[mask_bool].mean()),
        "per_neuron_mean_nll_per_masked_bin": float(per_neuron_nll[mask_bool].mean()),
        "per_time_neuron_mean_nll_per_masked_bin": float(per_time_neuron_nll[mask_bool].mean()),
        "model_mae_per_masked_bin": float(np.abs(pred[mask_bool] - y[mask_bool]).mean()),
        "model_mse_per_masked_bin": float(((pred[mask_bool] - y[mask_bool]) ** 2).mean()),
        "bits_per_spike_vs_per_neuron_mean": (
            float((test_null_sum - model_sum) / (math.log(2) * masked_spikes))
            if masked_spikes > 0 else float("nan")
        ),
        "corr_masked_bins": _safe_corr(y[mask_bool].reshape(-1), pred[mask_bool].reshape(-1)),
    }

    per_neuron_rows: List[Dict[str, float]] = []
    for neuron_idx in range(y.shape[2]):
        m = mask_bool[:, :, neuron_idx]
        spike_sum = float(y[:, :, neuron_idx][m].sum())
        model_sum_n = float(model_nll[:, :, neuron_idx][m].sum()) if m.any() else float("nan")
        test_null_sum_n = float(test_null_nll[:, :, neuron_idx][m].sum()) if m.any() else float("nan")
        per_neuron_rows.append({
            "neuron_idx": neuron_idx,
            "masked_bins": int(m.sum()),
            "masked_spikes": spike_sum,
            "model_nll_per_masked_bin": float(model_nll[:, :, neuron_idx][m].mean()) if m.any() else float("nan"),
            "per_neuron_mean_nll_per_masked_bin": float(per_neuron_nll[:, :, neuron_idx][m].mean()) if m.any() else float("nan"),
            "bits_per_spike_vs_per_neuron_mean": (
                float((test_null_sum_n - model_sum_n) / (math.log(2) * spike_sum))
                if m.any() and spike_sum > 0 else float("nan")
            ),
            "corr": _safe_corr(y[:, :, neuron_idx][m].reshape(-1), pred[:, :, neuron_idx][m].reshape(-1)) if m.any() else float("nan"),
            # window_r2 filled in below after _compute_window_r2
            "window_r2": float("nan"),
            "window_corr": float("nan"),
        })

    r2_per_neuron, r2_aggregate = _compute_window_r2(y, pred, mask_bool.astype(np.uint8))
    for row in per_neuron_rows:
        n = int(row["neuron_idx"])
        if n in r2_per_neuron:
            row["window_r2"] = r2_per_neuron[n]
            # Window-level Pearson r alongside R^2 
            m_n = mask_bool[:, :, n]
            if m_n.any():
                tw = (y[:, :, n] * m_n).sum(axis=1) / np.maximum(m_n.sum(axis=1), 1)
                pw = (pred[:, :, n] * m_n).sum(axis=1) / np.maximum(m_n.sum(axis=1), 1)
                valid = m_n.any(axis=1)
                row["window_corr"] = _safe_corr(tw[valid], pw[valid])

    summary.update(r2_aggregate)

    per_time_rows: List[Dict[str, float]] = []
    for time_idx in range(y.shape[1]):
        m = mask_bool[:, time_idx, :]
        per_time_rows.append({
            "time_idx": time_idx,
            "masked_bins": int(m.sum()),
            "model_nll_per_masked_bin": float(model_nll[:, time_idx, :][m].mean()) if m.any() else float("nan"),
            "per_neuron_mean_nll_per_masked_bin": float(per_neuron_nll[:, time_idx, :][m].mean()) if m.any() else float("nan"),
            "corr": _safe_corr(y[:, time_idx, :][m].reshape(-1), pred[:, time_idx, :][m].reshape(-1)) if m.any() else float("nan"),
        })

    return summary, per_neuron_rows, per_time_rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_eval_artifacts(
    artifact_dir: Path,
    targets: np.ndarray,
    pred_rates: np.ndarray,
    eval_mask: np.ndarray,
    per_neuron_rows: List[Dict[str, float]],
    per_time_rows: List[Dict[str, float]],
    scatter_points: int,
) -> None:
    _prepare_matplotlib_cache(artifact_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for evaluation plots: {exc}")
        return

    mask_bool = eval_mask.astype(bool)
    masked_y = targets[mask_bool].reshape(-1)
    masked_pred = pred_rates[mask_bool].reshape(-1)
    if masked_y.size:
        rng = np.random.default_rng(0)
        keep = rng.choice(masked_y.size, size=min(scatter_points, masked_y.size), replace=False)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(masked_y[keep], masked_pred[keep], s=5, alpha=0.25)
        max_axis = float(max(np.max(masked_y[keep]), np.max(masked_pred[keep]), 1.0))
        ax.plot([0, max_axis], [0, max_axis], color="black", linewidth=1.0)
        ax.set_xlabel("true held-out spike count")
        ax.set_ylabel("predicted rate")
        ax.set_title("Held-out bins: true vs predicted")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(artifact_dir / "true_vs_pred_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    example_idx = 0
    neuron_activity = (targets[example_idx] * eval_mask[example_idx]).sum(axis=0)
    plot_neurons = np.argsort(neuron_activity)[-min(48, targets.shape[2]):]
    plot_neurons = np.sort(plot_neurons)
    if plot_neurons.size:
        true_panel = targets[example_idx, :, plot_neurons].T
        pred_panel = pred_rates[example_idx, :, plot_neurons].T
        resid_panel = pred_panel - true_panel
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharey=True)
        panels = [(true_panel, "true spikes"), (pred_panel, "predicted rate"), (resid_panel, "pred - true")]
        for ax, (panel, title) in zip(axes, panels):
            im = ax.imshow(panel, aspect="auto", interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("time bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("selected neuron")
        fig.tight_layout()
        fig.savefig(artifact_dir / "example_reconstruction_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    valid_bps = [row["bits_per_spike_vs_per_neuron_mean"] for row in per_neuron_rows if not np.isnan(row["bits_per_spike_vs_per_neuron_mean"])]
    if valid_bps:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.hist(valid_bps, bins=40)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("bits/spike vs per-neuron mean")
        ax.set_ylabel("neuron count")
        ax.set_title("Per-neuron held-out improvement")
        fig.tight_layout()
        fig.savefig(artifact_dir / "per_neuron_bps_histogram.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    time_rows = [row for row in per_time_rows if row["masked_bins"] > 0]
    if time_rows:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot([row["time_idx"] for row in time_rows], [row["model_nll_per_masked_bin"] for row in time_rows], label="model")
        ax.plot([row["time_idx"] for row in time_rows], [row["per_neuron_mean_nll_per_masked_bin"] for row in time_rows], label="per-neuron mean")
        ax.set_xlabel("time bin")
        ax.set_ylabel("NLL per masked bin")
        ax.set_title("Held-out NLL by time")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(artifact_dir / "per_time_nll.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _compute_calibration_rows(
    targets: np.ndarray,
    pred_rates: np.ndarray,
    eval_mask: np.ndarray,
    baselines: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    
    mask_bool = eval_mask.astype(bool)
    flat_pred = pred_rates[mask_bool].reshape(-1).astype(np.float64)
    flat_target = targets[mask_bool].reshape(-1).astype(np.float64)
    flat_baseline = np.broadcast_to(baselines["per_neuron"][None, None, :], targets.shape)[mask_bool].reshape(-1).astype(np.float64)
    if flat_pred.size == 0:
        return []

    edges = np.quantile(flat_pred, np.linspace(0.0, 1.0, n_bins + 1))
    rows: List[Dict[str, float]] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            keep = (flat_pred >= edges[idx]) & (flat_pred <= edges[idx + 1])
        else:
            keep = (flat_pred >= edges[idx]) & (flat_pred < edges[idx + 1])
        if not np.any(keep):
            continue
        rows.append({
            "bin": float(idx),
            "n": float(np.sum(keep)),
            "pred_mean": float(np.mean(flat_pred[keep])),
            "target_mean": float(np.mean(flat_target[keep])),
            "baseline_mean": float(np.mean(flat_baseline[keep])),
            "nonzero_fraction": float(np.mean(flat_target[keep] > 0.0)),
        })
    return rows


def _compute_spike_count_group_rows(
    targets: np.ndarray,
    pred_rates: np.ndarray,
    eval_mask: np.ndarray,
    baselines: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
   
    mask_bool = eval_mask.astype(bool)
    model_nll = _poisson_nll_from_rate(pred_rates, targets)
    baseline_rate = np.broadcast_to(baselines["per_neuron"][None, None, :], targets.shape).astype(np.float64)
    baseline_nll = _poisson_nll_from_rate(baseline_rate, targets)

    groups = [
        ("0 spikes", targets == 0),
        ("1 spike", targets == 1),
        ("2+ spikes", targets >= 2),
        ("3+ spikes", targets >= 3),
        ("5+ spikes", targets >= 5),
    ]
    rows: List[Dict[str, float]] = []
    for label, condition in groups:
        keep = mask_bool & condition
        if not np.any(keep):
            continue
        rows.append({
            "group": label,
            "bins": float(np.sum(keep)),
            "spikes": float(np.sum(targets[keep])),
            "model_nll": float(np.mean(model_nll[keep])),
            "baseline_nll": float(np.mean(baseline_nll[keep])),
            "nll_gain": float(np.mean(baseline_nll[keep] - model_nll[keep])),
            "pred_mean": float(np.mean(pred_rates[keep])),
            "target_mean": float(np.mean(targets[keep])),
        })
    return rows


def _plot_calibration_artifact(
    artifact_dir: Path,
    calibration_rows: List[Dict[str, float]],
) -> None:
    if not calibration_rows:
        return
    _prepare_matplotlib_cache(artifact_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for calibration plot: {exc}")
        return

    pred_mean = np.asarray([row["pred_mean"] for row in calibration_rows], dtype=np.float64)
    target_mean = np.asarray([row["target_mean"] for row in calibration_rows], dtype=np.float64)
    baseline_mean = np.asarray([row["baseline_mean"] for row in calibration_rows], dtype=np.float64)
    nonzero_fraction = np.asarray([row["nonzero_fraction"] for row in calibration_rows], dtype=np.float64)
    max_axis = max(float(np.nanmax(pred_mean)), float(np.nanmax(target_mean)), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(pred_mean, target_mean, marker="o", label="model calibration")
    axes[0].plot(pred_mean, baseline_mean, marker="o", label="baseline mean")
    axes[0].plot([0.0, max_axis], [0.0, max_axis], color="black", linewidth=1.0, label="ideal")
    axes[0].set_xlabel("mean predicted rate in decile")
    axes[0].set_ylabel("mean true count")
    axes[0].set_title("Calibration by predicted rate")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(pred_mean, nonzero_fraction, marker="o")
    axes[1].set_xlabel("mean predicted rate in decile")
    axes[1].set_ylabel("fraction nonzero")
    axes[1].set_title("Spike-event enrichment")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(artifact_dir / "calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_spike_count_group_artifact(
    artifact_dir: Path,
    spike_count_rows: List[Dict[str, float]],
) -> None:
    if not spike_count_rows:
        return
    _prepare_matplotlib_cache(artifact_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for spike-count diagnostics: {exc}")
        return

    labels = [row["group"] for row in spike_count_rows]
    model = np.asarray([row["model_nll"] for row in spike_count_rows], dtype=np.float64)
    baseline = np.asarray([row["baseline_nll"] for row in spike_count_rows], dtype=np.float64)
    gain = np.asarray([row["nll_gain"] for row in spike_count_rows], dtype=np.float64)
    x = np.arange(len(spike_count_rows))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(x - width / 2.0, model, width=width, label="model")
    axes[0].bar(x + width / 2.0, baseline, width=width, label="per-neuron baseline")
    axes[0].set_ylabel("NLL per masked bin")
    axes[0].set_title("Loss by true spike count")
    axes[0].legend(frameon=False)

    axes[1].bar(x, gain)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("baseline - model NLL")
    axes[1].set_title("Gain by true spike count")

    for ax in axes:
        ax.set_xticks(x, labels=labels, rotation=25, ha="right")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(artifact_dir / "spike_count_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_split_eval_summary(save_dir: str, split_summaries: Dict[str, Dict[str, Any]]) -> None:
    artifact_dir = _artifact_dir(save_dir)
    with open(artifact_dir / "split_eval_summary.json", "w") as f:
        json.dump(_json_ready(split_summaries), f, indent=2)

    csv_rows: List[Dict[str, Any]] = []
    for split_name, metrics in split_summaries.items():
        row = {"split": split_name}
        row.update(metrics)
        csv_rows.append(row)
    _write_csv(artifact_dir / "split_eval_summary.csv", csv_rows)

    if "train" in split_summaries and "test" in split_summaries:
        train_metrics = split_summaries["train"]
        test_metrics = split_summaries["test"]
        gap_payload = {
            "train_model_nll_per_masked_bin": train_metrics.get("model_nll_per_masked_bin"),
            "test_model_nll_per_masked_bin": test_metrics.get("model_nll_per_masked_bin"),
            "test_minus_train_model_nll_per_masked_bin": (
                test_metrics.get("model_nll_per_masked_bin", float("nan"))
                - train_metrics.get("model_nll_per_masked_bin", float("nan"))
            ),
            "train_bits_per_spike_vs_per_neuron_mean": train_metrics.get("bits_per_spike_vs_per_neuron_mean"),
            "test_bits_per_spike_vs_per_neuron_mean": test_metrics.get("bits_per_spike_vs_per_neuron_mean"),
            "test_minus_train_bits_per_spike_vs_per_neuron_mean": (
                test_metrics.get("bits_per_spike_vs_per_neuron_mean", float("nan"))
                - train_metrics.get("bits_per_spike_vs_per_neuron_mean", float("nan"))
            ),
        }
        with open(artifact_dir / "generalization_gap.json", "w") as f:
            json.dump(_json_ready(gap_payload), f, indent=2)


def _save_latent_pca(
    artifact_dir: Path,
    latent_rows: List[np.ndarray],
    latent_sequence_rows: Optional[List[np.ndarray]] = None,
    prefix: str = "latent",
) -> None:
    if not latent_rows:
        return
    if latent_sequence_rows:
        np.savez_compressed(
            artifact_dir / f"{prefix}_sequence.npz",
            latents=np.concatenate(latent_sequence_rows, axis=0).astype(np.float32),
        )
    latents = np.concatenate(latent_rows, axis=0).astype(np.float64)
    np.save(artifact_dir / f"{prefix}_mean_pooled.npy", latents.astype(np.float32))
    latent_mean = latents.mean(axis=0, keepdims=True)
    centered = latents - latent_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, vt.shape[0])
    components = vt[:n_components]
    coords = centered @ components.T
    if n_components < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - n_components)), mode="constant")
        components = np.pad(components, ((0, 2 - n_components), (0, 0)), mode="constant")
        singular_values = np.pad(singular_values[:n_components], (0, 2 - n_components), mode="constant")
    else:
        singular_values = singular_values[:2]
    np.savez_compressed(
        artifact_dir / f"{prefix}_pca_basis.npz",
        coords=coords.astype(np.float32),
        mean=latent_mean.astype(np.float32),
        components=components.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
    )
    _write_csv(
        artifact_dir / f"{prefix}_pca.csv",
        [{"example_idx": int(i), "pc1": float(x), "pc2": float(y)} for i, (x, y) in enumerate(coords)],
    )

    _prepare_matplotlib_cache(artifact_dir)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] could not import matplotlib for latent PCA plot: {exc}")
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(coords.shape[0]), s=12, cmap="viridis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{prefix.replace('_', ' ')} PCA")
    fig.colorbar(scatter, ax=ax, label="eval example order")
    fig.tight_layout()
    fig.savefig(artifact_dir / f"{prefix}_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save_artifacts(
    model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    masking_mode: str,
    eval_split: str,
    mask_ratio: float,
    baselines: Dict[str, np.ndarray],
    use_lograte: bool,
    eval_seed: int,
    max_batches: Optional[int] = None,
    save_latents: bool = True,
    scatter_points: int = 5000,
    artifact_subdir: Optional[str] = None,
) -> Dict[str, float]:
    # Run deterministic held-out evaluation and save all report inputs in one place.
    artifact_dir = _artifact_dir(save_dir, artifact_subdir)
    rng = np.random.default_rng(eval_seed)
    cached_selection: Dict[str, np.ndarray] = {}
    old_encoder_mask = getattr(model.encoder, "mask", None)

    targets_list: List[np.ndarray] = []
    pred_rates_list: List[np.ndarray] = []
    masked_inputs_list: List[np.ndarray] = []
    eval_masks_list: List[np.ndarray] = []
    latent_rows: List[np.ndarray] = []
    latent_sequence_rows: List[np.ndarray] = []
    unmasked_latent_rows: List[np.ndarray] = []
    unmasked_latent_sequence_rows: List[np.ndarray] = []
    latest_latent: Dict[str, Optional[torch.Tensor]] = {"value": None}

    def _latent_hook(_module, _inputs, output):
        latest_latent["value"] = output.detach()

    def _record_latest_latent(mean_rows: List[np.ndarray], sequence_rows: List[np.ndarray], original_length: int) -> None:
        if latest_latent["value"] is None:
            return
        latent = latest_latent["value"]
        if latent.shape[1] > original_length:
            latent = latent[:, -original_length:, :]
        latent_np = latent.detach().cpu().numpy().astype(np.float32)
        sequence_rows.append(latent_np)
        mean_rows.append(latent_np.mean(axis=1))

    hook = model.encoder.out_proj.register_forward_hook(_latent_hook) if save_latents else None
    model.eval()
    if old_encoder_mask is not None:
        model.encoder.mask = False

    try:
        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                if max_batches is not None and step >= max_batches:
                    break

                batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                original_spikes = batch["spikes_data"].clone()
                eval_mask_cpu, cached_selection = _make_eval_mask(
                    tuple(original_spikes.shape), masking_mode, mask_ratio, rng, cached_selection
                )
                eval_mask = eval_mask_cpu.to(device)
                masked_spikes = original_spikes.clone()
                masked_spikes[eval_mask.bool()] = 0

                latest_latent["value"] = None
                outputs = model(
                    masked_spikes,
                    time_attn_mask=batch["time_attn_mask"],
                    space_attn_mask=batch["space_attn_mask"],
                    spikes_timestamps=batch["spikes_timestamps"],
                    spikes_spacestamps=batch["spikes_spacestamps"],
                    targets=batch["target"],
                    neuron_regions=batch["neuron_regions"],
                    eval_mask=eval_mask,
                    masking_mode=masking_mode,
                    spike_augmentation=False,
                    num_neuron=masked_spikes.shape[2],
                    eid=batch["eid"][0],
                )

                preds = outputs.preds.detach()
                if use_lograte:
                    pred_rates = torch.exp(torch.clamp(preds, min=-20.0, max=20.0))
                else:
                    pred_rates = torch.clamp(preds, min=1e-9)

                if save_latents and latest_latent["value"] is not None:
                    # Store masked-evaluation latents alongside predictions for reconstruction analysis.
                    _record_latest_latent(latent_rows, latent_sequence_rows, original_spikes.shape[1])

                    latest_latent["value"] = None
                    zero_eval_mask = torch.zeros_like(eval_mask)
                    _ = model(
                        original_spikes,
                        time_attn_mask=batch["time_attn_mask"],
                        space_attn_mask=batch["space_attn_mask"],
                        spikes_timestamps=batch["spikes_timestamps"],
                        spikes_spacestamps=batch["spikes_spacestamps"],
                        targets=batch["target"],
                        neuron_regions=batch["neuron_regions"],
                        eval_mask=zero_eval_mask,
                        masking_mode=masking_mode,
                        spike_augmentation=False,
                        num_neuron=original_spikes.shape[2],
                        eid=batch["eid"][0],
                    )
                    _record_latest_latent(unmasked_latent_rows, unmasked_latent_sequence_rows, original_spikes.shape[1])

                targets_list.append(original_spikes.detach().cpu().numpy().astype(np.float32))
                pred_rates_list.append(pred_rates.detach().cpu().numpy().astype(np.float32))
                masked_inputs_list.append(masked_spikes.detach().cpu().numpy().astype(np.float32))
                eval_masks_list.append(eval_mask.detach().cpu().numpy().astype(np.uint8))
    finally:
        if old_encoder_mask is not None:
            model.encoder.mask = old_encoder_mask
        if hook is not None:
            hook.remove()

    if not targets_list:
        raise RuntimeError("No evaluation batches were processed; cannot save artifacts.")

    targets = np.concatenate(targets_list, axis=0)
    pred_rates = np.concatenate(pred_rates_list, axis=0)
    masked_inputs = np.concatenate(masked_inputs_list, axis=0)
    eval_mask = np.concatenate(eval_masks_list, axis=0)

    summary, per_neuron_rows, per_time_rows = _summarise_metrics(targets, pred_rates, eval_mask, baselines)
    summary.update({
        "masking_mode": masking_mode,
        "eval_split": eval_split,
        "eval_mask_ratio": float(mask_ratio),
        "eval_examples": int(targets.shape[0]),
    })

    with open(artifact_dir / "eval_metrics.json", "w") as f:
        json.dump(_json_ready(summary), f, indent=2)
    _write_csv(artifact_dir / "eval_metrics.csv", [summary])
    _write_csv(artifact_dir / "per_neuron_metrics.csv", per_neuron_rows)
    _write_csv(artifact_dir / "per_time_metrics.csv", per_time_rows)

    np.savez_compressed(
        artifact_dir / "eval_predictions.npz",
        targets=targets,
        pred_rates=pred_rates,
        masked_inputs=masked_inputs,
        eval_mask=eval_mask,
        eval_dataset_indices=np.asarray(
            getattr(getattr(eval_loader, "dataset", None), "indices", np.asarray([], dtype=np.int64))
        )[:targets.shape[0]],
        baseline_global=np.asarray(baselines["global"]),
        baseline_per_neuron=baselines["per_neuron"],
        baseline_per_time_neuron=baselines["per_time_neuron"],
        heldout_neuron_indices=cached_selection.get("neuron_indices", np.asarray([], dtype=np.int64)),
        heldout_time_indices=cached_selection.get("time_indices", np.asarray([], dtype=np.int64)),
    )

    calibration_rows = _compute_calibration_rows(targets, pred_rates, eval_mask, baselines)
    spike_count_rows = _compute_spike_count_group_rows(targets, pred_rates, eval_mask, baselines)
    _write_csv(artifact_dir / "calibration.csv", calibration_rows)
    _write_csv(artifact_dir / "spike_count_groups.csv", spike_count_rows)

    _plot_eval_artifacts(artifact_dir, targets, pred_rates, eval_mask, per_neuron_rows, per_time_rows, scatter_points)
    _plot_calibration_artifact(artifact_dir, calibration_rows)
    _plot_spike_count_group_artifact(artifact_dir, spike_count_rows)
    if save_latents:
        _save_latent_pca(artifact_dir, latent_rows, latent_sequence_rows)
        _save_latent_pca(
            artifact_dir,
            unmasked_latent_rows,
            unmasked_latent_sequence_rows,
            prefix="unmasked_latent",
        )

    print(f"[INFO] saved evaluation artifacts to: {artifact_dir}")
    print(f"[INFO] eval model NLL/bin: {summary['model_nll_per_masked_bin']:.8f}")
    print(f"[INFO] eval per-neuron baseline NLL/bin: {summary['per_neuron_mean_nll_per_masked_bin']:.8f}")
    print(f"[INFO] eval bits/spike vs per-neuron baseline: {summary['bits_per_spike_vs_per_neuron_mean']:.8f}")
    return summary
