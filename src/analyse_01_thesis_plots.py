

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunDef:
    run_name: str
    label: str
    color: str
    line_style: str = "-"


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    title: str
    source_type: str
    eval_modes: Tuple[str, ...]
    runs: Tuple[RunDef, ...]
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RunBundle:
    run_name: str
    legend_label: str
    title_label: str
    color: str
    line_style: str
    eval_mode: str
    source_type: str
    params: Dict[str, object]
    metrics: Dict[str, object]
    per_neuron: pd.DataFrame
    per_time: pd.DataFrame


PALETTE = {
    "mtm_plain": "#1F77B4",
    "mtm_adapter": "#0F8C8C",
    "scratch": "#D62728",
    "stitched": "#2CA02C",
    "wd_high": "#FF7F0E",
    "lr_low": "#9467BD",
    "lr_high": "#E15759",
    "adapter_low": "#76B7B2",
    "scale_up": "#8C564B",
    "rerun": "#B07AA1",
    "recipe": "#BCBD22",
}

METRIC_INFO = {
    "window_r2": {"title": "Window R2", "ylabel": "Window R2", "zero_line": True},
    "nll_gain": {"title": "NLL gain", "ylabel": "Baseline NLL - model NLL", "zero_line": True},
    "bits_per_spike_vs_per_neuron_mean": {"title": "Bits / spike", "ylabel": "Bits / spike", "zero_line": True},
    "corr": {"title": "Pearson correlation", "ylabel": "Correlation", "zero_line": True},
}

TIME_METRIC_INFO = {
    "model_nll_per_masked_bin": {"title": "Model NLL over time", "ylabel": "Model NLL / masked bin"},
    "nll_gain": {"title": "NLL gain over time", "ylabel": "Baseline NLL - model NLL"},
    "corr": {"title": "Correlation over time", "ylabel": "Correlation"},
}

COMPARISONS: Tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        key="neuron_adapter_effect",
        title="Neuron Terms: Adapter vs Plain IBL-MtM",
        source_type="single",
        eval_modes=("neuron",),
        runs=(
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("local_neuron_mask_full_plain_lr2e-5", "IBL-MtM plain", PALETTE["mtm_plain"]),
        ),
    ),
    ComparisonSpec(
        key="neuron_vs_scratch",
        title="Neuron Terms: IBL-MtM vs NDT1 Scratch",
        source_type="single",
        eval_modes=("neuron",),
        runs=(
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_neuron_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
        ),
    ),
    ComparisonSpec(
        key="neuron_vs_stitched",
        title="Neuron Terms: IBL-MtM vs Stitched NDT1",
        source_type="single",
        eval_modes=("neuron",),
        runs=(
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_stitched_neuron_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="neuron_final_summary",
        title="Neuron Terms: Final Summary",
        source_type="single",
        eval_modes=("neuron",),
        runs=(
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_neuron_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
            RunDef("ndt1_stitched_neuron_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="causal_adapter_effect",
        title="Causal Terms: Adapter vs Plain IBL-MtM",
        source_type="single",
        eval_modes=("causal",),
        runs=(
            RunDef("causal_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("causal_full_plain_lr2e-5", "IBL-MtM plain", PALETTE["mtm_plain"]),
        ),
    ),
    ComparisonSpec(
        key="causal_vs_scratch",
        title="Causal Terms: IBL-MtM vs NDT1 Scratch",
        source_type="single",
        eval_modes=("causal",),
        runs=(
            RunDef("causal_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_causal_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
        ),
    ),
    ComparisonSpec(
        key="causal_vs_stitched",
        title="Causal Terms: IBL-MtM vs Stitched NDT1",
        source_type="single",
        eval_modes=("causal",),
        runs=(
            RunDef("causal_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_stitched_causal_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="causal_final_summary",
        title="Causal Terms: Final Summary",
        source_type="single",
        eval_modes=("causal",),
        runs=(
            RunDef("causal_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_causal_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
            RunDef("ndt1_stitched_causal_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="combined_adapter_effect",
        title="Combined Models: Adapter vs Plain IBL-MtM",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_plain_lr2e-5", "IBL-MtM plain", PALETTE["mtm_plain"]),
        ),
    ),
    ComparisonSpec(
        key="combined_vs_scratch",
        title="Combined Models: IBL-MtM vs NDT1 Scratch",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_combined_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
        ),
    ),
    ComparisonSpec(
        key="combined_vs_stitched",
        title="Combined Models: IBL-MtM vs Stitched NDT1",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_stitched_combined_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="combined_final_summary",
        title="Combined Models: Final Summary",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "IBL-MtM adapter", PALETTE["mtm_adapter"]),
            RunDef("ndt1_direct_combined_full_lr2e-5", "NDT1 scratch", PALETTE["scratch"]),
            RunDef("ndt1_stitched_combined_full_prompt_lr2e-5", "Stitched NDT1", PALETTE["stitched"]),
        ),
    ),
    ComparisonSpec(
        key="neuron_regularisation_change",
        title="Neuron Terms: Weight Decay Change Within IBL-MtM",
        source_type="single",
        eval_modes=("neuron",),
        runs=(
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "wd 0.01", PALETTE["mtm_adapter"]),
            RunDef("local_neuron_mask_full_lr2e-5_adapter5e-5", "wd 0.1", PALETTE["wd_high"]),
        ),
    ),
    ComparisonSpec(
        key="causal_scale_regularisation_change",
        title="Causal Terms: Training Scale and Weight Decay Change",
        source_type="single",
        eval_modes=("causal",),
        runs=(
            RunDef("causal_full_lr2e-5_adapter5e-5", "bs 8 | ep 100 | wd 0.01", PALETTE["mtm_adapter"]),
            RunDef("causal_full_lr2e-5_adapter5e-5", "bs 16 | ep 150 | wd 0.1", PALETTE["wd_high"]),
        ),
    ),
    ComparisonSpec(
        key="combined_scale_up",
        title="Combined Models: bs 8 / 100 epochs vs bs 16 / 150 epochs",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "bs 8 | ep 100", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5", "bs 16 | ep 150", PALETTE["scale_up"]),
        ),
    ),
    ComparisonSpec(
        key="combined_lr_low",
        title="Combined Models: Backbone LR 2e-5 vs 1e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "lr 2e-5 | warm-up 0.20", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr1e-5_adapter5e-5_rerun", "lr 1e-5 | warm-up 0.20", PALETTE["lr_low"]),
        ),
    ),
    ComparisonSpec(
        key="combined_lr_high",
        title="Combined Models: Backbone LR 2e-5 vs 3e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "lr 2e-5 | warm-up 0.20", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun", "lr 3e-5 | warm-up 0.20", PALETTE["lr_high"]),
        ),
    ),
    ComparisonSpec(
        key="combined_adapter_lr",
        title="Combined Models: Adapter LR 5e-5 vs 3e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "adapter lr 5e-5", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter3e-5_rerun", "adapter lr 3e-5", PALETTE["adapter_low"]),
        ),
    ),
    ComparisonSpec(
        key="combined_weight_decay_2e5",
        title="Combined Models: Weight Decay 0.01 vs 0.1 at 2e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd001", "wd 0.01 | warm-up 0.15", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd01", "wd 0.1 | warm-up 0.15", PALETTE["wd_high"]),
        ),
    ),
    ComparisonSpec(
        key="combined_weight_decay_3e5",
        title="Combined Models: Weight Decay 0.01 vs 0.1 at 3e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd001", "wd 0.01 | warm-up 0.15", PALETTE["lr_high"]),
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd01", "wd 0.1 | warm-up 0.15", PALETTE["wd_high"]),
        ),
    ),
    ComparisonSpec(
        key="combined_warmup_2e5_wd001",
        title="Combined Models: Warm-up 0.15 vs 0.20 at 2e-5, wd 0.01",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd001", "warm-up 0.15", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "warm-up 0.20", PALETTE["rerun"], line_style="--"),
        ),
    ),
    ComparisonSpec(
        key="combined_warmup_3e5_wd001",
        title="Combined Models: Warm-up 0.15 vs 0.20 at 3e-5, wd 0.01",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd001", "warm-up 0.15", PALETTE["lr_high"]),
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun", "warm-up 0.20", PALETTE["rerun"], line_style="--"),
        ),
    ),
    ComparisonSpec(
        key="combined_warmup_3e5_wd01",
        title="Combined Models: Warm-up 0.15 vs 0.20 at 3e-5, wd 0.1",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_true015_3e5_wd01", "warm-up 0.15", PALETTE["wd_high"]),
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun", "warm-up 0.20", PALETTE["rerun"], line_style="--"),
        ),
    ),
    ComparisonSpec(
        key="combined_recipe_shift_2e5",
        title="Combined Models: 0.01 / 0.15 vs 0.1 / 0.20 at 2e-5",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_true015_2e5_wd001", "wd 0.01 | warm-up 0.15", PALETTE["mtm_adapter"]),
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "wd 0.1 | warm-up 0.20", PALETTE["recipe"], line_style="--"),
        ),
    ),
    ComparisonSpec(
        key="combined_recipe_shift_high_lr",
        title="Combined Models: 2e-5 / wd 0.1 / wu 0.20 vs 3e-5 / wd 0.1 / wu 0.20",
        source_type="combined",
        eval_modes=("neuron", "causal"),
        runs=(
            RunDef("ibl_mtm_combined_direct_full_lr2e-5_adapter5e-5_rerun", "lr 2e-5", PALETTE["recipe"], line_style="--"),
            RunDef("ibl_mtm_combined_direct_full_lr3e-5_adapter5e-5_rerun", "lr 3e-5", PALETTE["lr_high"], line_style="--"),
        ),
    ),
)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def _read_json(path: Path) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def _family_label(run_name: str, eval_mode: str) -> str:
    if run_name.startswith("local_neuron_mask_full_"):
        return "IBL-MtM neuron"
    if run_name.startswith("causal_full_"):
        return "IBL-MtM causal"
    if run_name.startswith("ibl_mtm_combined_direct_"):
        return f"IBL-MtM combined ({eval_mode})"
    if run_name.startswith("ndt1_direct_neuron_"):
        return "NDT1 scratch neuron"
    if run_name.startswith("ndt1_direct_causal_"):
        return "NDT1 scratch causal"
    if run_name.startswith("ndt1_direct_combined_"):
        return f"NDT1 scratch combined ({eval_mode})"
    if run_name.startswith("ndt1_stitched_neuron_"):
        return "Stitched NDT1 neuron"
    if run_name.startswith("ndt1_stitched_causal_"):
        return "Stitched NDT1 causal"
    if run_name.startswith("ndt1_stitched_combined_"):
        return f"Stitched NDT1 combined ({eval_mode})"
    return run_name


def _float_or_dash(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "-"
    return f"{float(value):g}"


def _extract_params(run_name: str, metadata: Dict[str, object]) -> Dict[str, object]:
    args = metadata.get("args", {})
    config = metadata.get("config", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    return {
        "lr": args.get("lr", optimizer.get("lr")),
        "adapter_lr": args.get("adapter_lr"),
        "weight_decay": args.get("weight_decay", optimizer.get("wd")),
        "warmup_pct": args.get("warmup_pct", optimizer.get("warmup_pct")),
        "batch_size": args.get("batch_size", training.get("train_batch_size")),
        "epochs": args.get("epochs", training.get("num_epochs")),
    }


def _title_with_params(run_name: str, eval_mode: str, params: Dict[str, object]) -> str:
    family = _family_label(run_name, eval_mode)
    parts = [family]
    if params.get("lr") is not None:
        parts.append(f"lr {params['lr']:g}")
    if params.get("adapter_lr") is not None:
        parts.append(f"adapter {params['adapter_lr']:g}")
    if params.get("weight_decay") is not None:
        parts.append(f"wd {params['weight_decay']:g}")
    if params.get("warmup_pct") is not None:
        parts.append(f"warm-up {params['warmup_pct']:g}")
    return " | ".join(parts)


def _artifact_root(results_dir: Path, run_name: str, source_type: str, eval_mode: str) -> Path:
    root = results_dir / run_name / "artifacts"
    if source_type == "combined":
        root = root / f"eval_test_{eval_mode}"
    return root


def _prepare_bundle(results_dir: Path, run_def: RunDef, source_type: str, eval_mode: str) -> RunBundle:
    run_dir = results_dir / run_def.run_name
    metadata = _read_json(run_dir / "artifacts" / "run_metadata.json")
    params = _extract_params(run_def.run_name, metadata)

    artifact_root = _artifact_root(results_dir, run_def.run_name, source_type, eval_mode)
    eval_metrics = _read_json(artifact_root / "eval_metrics.json")
    per_neuron = pd.read_csv(artifact_root / "per_neuron_metrics.csv")
    per_time = pd.read_csv(artifact_root / "per_time_metrics.csv")

    per_neuron["nll_gain"] = (
        per_neuron["per_neuron_mean_nll_per_masked_bin"] - per_neuron["model_nll_per_masked_bin"]
    )
    per_time["nll_gain"] = (
        per_time["per_neuron_mean_nll_per_masked_bin"] - per_time["model_nll_per_masked_bin"]
    )

    return RunBundle(
        run_name=run_def.run_name,
        legend_label=run_def.label,
        title_label=_title_with_params(run_def.run_name, eval_mode, params),
        color=run_def.color,
        line_style=run_def.line_style,
        eval_mode=eval_mode,
        source_type=source_type,
        params=params,
        metrics=eval_metrics,
        per_neuron=per_neuron,
        per_time=per_time,
    )


def _series_clean(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].replace([np.inf, -np.inf], np.nan).dropna()


def _plot_individual_histograms(bundle: RunBundle, output_path: Path, plt) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    columns = ["window_r2", "nll_gain", "bits_per_spike_vs_per_neuron_mean", "corr"]

    for ax, column in zip(axes.flat, columns):
        series = _series_clean(bundle.per_neuron, column)
        if series.empty:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(series.to_numpy(), bins=28, color=bundle.color, alpha=0.85, edgecolor="white")
        ax.set_title(METRIC_INFO[column]["title"], fontsize=11, pad=10)
        ax.set_xlabel(METRIC_INFO[column]["ylabel"])
        ax.set_ylabel("Neuron count")
        ax.grid(True, alpha=0.22)
        if METRIC_INFO[column]["zero_line"]:
            ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)

    fig.suptitle(bundle.title_label, fontsize=14, y=0.98)
    fig.text(0.5, 0.945, "Per-neuron metric distributions", ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98, hspace=0.34, wspace=0.22)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_individual_time_lines(bundle: RunBundle, output_path: Path, plt) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)
    time = bundle.per_time["time_idx"].to_numpy()

    axes[0].plot(time, bundle.per_time["model_nll_per_masked_bin"], color=bundle.color, linewidth=2.0, label="model")
    axes[0].plot(
        time,
        bundle.per_time["per_neuron_mean_nll_per_masked_bin"],
        color="#7F7F7F",
        linewidth=1.8,
        linestyle="--",
        label="baseline",
    )
    axes[0].set_ylabel("NLL / masked bin")
    axes[0].set_title("Model and baseline NLL over time", fontsize=11, pad=10)
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(time, bundle.per_time["nll_gain"], color=bundle.color, linewidth=2.0)
    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("NLL gain")
    axes[1].set_title("NLL gain over time", fontsize=11, pad=10)
    axes[1].grid(True, alpha=0.22)

    axes[2].plot(time, bundle.per_time["corr"], color=bundle.color, linewidth=2.0)
    axes[2].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_ylabel("Correlation")
    axes[2].set_xlabel("Time bin")
    axes[2].set_title("Correlation over time", fontsize=11, pad=10)
    axes[2].grid(True, alpha=0.22)

    fig.suptitle(bundle.title_label, fontsize=14, y=0.985)
    fig.text(0.5, 0.955, "Per-time diagnostics", ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.08, right=0.98, hspace=0.38)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _merge_metric(bundles: Sequence[RunBundle], frame_name: str, metric: str, index_col: str) -> Optional[pd.DataFrame]:
    merged: Optional[pd.DataFrame] = None
    for idx, bundle in enumerate(bundles):
        frame = getattr(bundle, frame_name)[[index_col, metric]].copy()
        frame = frame.rename(columns={metric: f"run_{idx}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=index_col, how="inner")

    if merged is None:
        return None

    value_cols = [c for c in merged.columns if c.startswith("run_")]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=value_cols)
    if merged.empty:
        return None
    return merged


def _plot_comparison_per_neuron_lines(spec: ComparisonSpec, eval_mode: str, bundles: Sequence[RunBundle], output_path: Path, plt) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    metric_order = ["window_r2", "nll_gain", "bits_per_spike_vs_per_neuron_mean", "corr"]

    for ax, metric in zip(axes.flat, metric_order):
        merged = _merge_metric(bundles, "per_neuron", metric, "neuron_idx")
        if merged is None:
            ax.text(0.5, 0.5, "No shared valid neurons", ha="center", va="center", transform=ax.transAxes)
            continue

        order = np.argsort(merged["run_0"].to_numpy())[::-1]
        x = np.arange(1, len(order) + 1)

        for idx, bundle in enumerate(bundles):
            y = merged[f"run_{idx}"].to_numpy()[order]
            ax.plot(
                x,
                y,
                color=bundle.color,
                linewidth=2.4 if bundle.line_style != "-" else 2.1,
                linestyle=bundle.line_style,
                alpha=0.95,
                label=bundle.legend_label,
            )

        if METRIC_INFO[metric]["zero_line"]:
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(METRIC_INFO[metric]["title"], fontsize=11, pad=10)
        ax.set_xlabel(f"Neuron rank (sorted by {bundles[0].legend_label})")
        ax.set_ylabel(METRIC_INFO[metric]["ylabel"])
        ax.grid(True, alpha=0.22)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=min(len(labels), 3), frameon=False)
    fig.suptitle(f"{spec.title} | {eval_mode.title()} evaluation", fontsize=14, y=0.985)
    fig.text(0.5, 0.955, "Per-neuron comparison curves", ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.89, bottom=0.12, left=0.07, right=0.98, hspace=0.34, wspace=0.24)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_per_neuron_deltas(spec: ComparisonSpec, eval_mode: str, bundles: Sequence[RunBundle], output_path: Path, plt) -> None:
    if len(bundles) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    metric_order = ["window_r2", "nll_gain", "bits_per_spike_vs_per_neuron_mean", "corr"]

    for ax, metric in zip(axes.flat, metric_order):
        merged = _merge_metric(bundles, "per_neuron", metric, "neuron_idx")
        if merged is None:
            ax.text(0.5, 0.5, "No shared valid neurons", ha="center", va="center", transform=ax.transAxes)
            continue

        order = np.argsort(merged["run_0"].to_numpy())[::-1]
        x = np.arange(1, len(order) + 1)
        ref = merged["run_0"].to_numpy()[order]

        for idx, bundle in enumerate(bundles[1:], start=1):
            delta = merged[f"run_{idx}"].to_numpy()[order] - ref
            ax.plot(
                x,
                delta,
                color=bundle.color,
                linewidth=2.4 if bundle.line_style != "-" else 2.0,
                linestyle=bundle.line_style,
                alpha=0.95,
                label=f"{bundle.legend_label} - {bundles[0].legend_label}",
            )

        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{METRIC_INFO[metric]['title']} delta", fontsize=11, pad=10)
        ax.set_xlabel(f"Neuron rank (sorted by {bundles[0].legend_label})")
        ax.set_ylabel("Difference")
        ax.grid(True, alpha=0.22)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=1 if len(labels) == 1 else 2, frameon=False)
    fig.suptitle(f"{spec.title} | {eval_mode.title()} evaluation", fontsize=14, y=0.985)
    fig.text(0.5, 0.955, "Per-neuron deltas vs reference", ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.89, bottom=0.13, left=0.07, right=0.98, hspace=0.34, wspace=0.24)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_per_time_lines(spec: ComparisonSpec, eval_mode: str, bundles: Sequence[RunBundle], output_path: Path, plt) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 10.5), sharex=True)
    baseline_drawn = False

    for bundle in bundles:
        time = bundle.per_time["time_idx"].to_numpy()
        axes[0].plot(time, bundle.per_time["model_nll_per_masked_bin"], color=bundle.color, linewidth=2.4 if bundle.line_style != "-" else 2.0, linestyle=bundle.line_style, alpha=0.95, label=bundle.legend_label)
        axes[1].plot(time, bundle.per_time["nll_gain"], color=bundle.color, linewidth=2.4 if bundle.line_style != "-" else 2.0, linestyle=bundle.line_style, alpha=0.95, label=bundle.legend_label)
        axes[2].plot(time, bundle.per_time["corr"], color=bundle.color, linewidth=2.4 if bundle.line_style != "-" else 2.0, linestyle=bundle.line_style, alpha=0.95, label=bundle.legend_label)

        if not baseline_drawn:
            axes[0].plot(
                time,
                bundle.per_time["per_neuron_mean_nll_per_masked_bin"],
                color="#7F7F7F",
                linewidth=1.8,
                linestyle="--",
                label="baseline",
            )
            baseline_drawn = True

    axes[0].set_ylabel("NLL / masked bin")
    axes[0].set_title("Model NLL over time", fontsize=11, pad=10)
    axes[0].grid(True, alpha=0.22)

    axes[1].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("NLL gain")
    axes[1].set_title("NLL gain over time", fontsize=11, pad=10)
    axes[1].grid(True, alpha=0.22)

    axes[2].axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_ylabel("Correlation")
    axes[2].set_xlabel("Time bin")
    axes[2].set_title("Correlation over time", fontsize=11, pad=10)
    axes[2].grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(f"{spec.title} | {eval_mode.title()} evaluation", fontsize=14, y=0.985)
    fig.text(0.5, 0.955, "Per-time comparison curves", ha="center", va="center", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.9, bottom=0.12, left=0.08, right=0.98, hspace=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bundle_summary_row(bundle: RunBundle, reference: Optional[RunBundle] = None) -> Dict[str, object]:
    metrics = bundle.metrics
    baseline_nll = metrics.get("per_neuron_mean_nll_per_masked_bin")
    model_nll = metrics.get("model_nll_per_masked_bin")
    nll_gain = None
    if baseline_nll is not None and model_nll is not None:
        nll_gain = float(baseline_nll) - float(model_nll)

    row = {
        "label": bundle.legend_label,
        "run_name": bundle.run_name,
        "eval_mode": bundle.eval_mode,
        "lr": _float_or_dash(bundle.params.get("lr")),
        "adapter_lr": _float_or_dash(bundle.params.get("adapter_lr")),
        "weight_decay": _float_or_dash(bundle.params.get("weight_decay")),
        "warmup_pct": _float_or_dash(bundle.params.get("warmup_pct")),
        "batch_size": bundle.params.get("batch_size"),
        "epochs": bundle.params.get("epochs"),
        "model_nll_per_masked_bin": _float_or_dash(model_nll),
        "baseline_nll_per_masked_bin": _float_or_dash(baseline_nll),
        "nll_gain": _float_or_dash(nll_gain),
        "bits_per_spike_vs_per_neuron_mean": _float_or_dash(metrics.get("bits_per_spike_vs_per_neuron_mean")),
        "corr_masked_bins": _float_or_dash(metrics.get("corr_masked_bins")),
        "median_window_r2": _float_or_dash(metrics.get("median_window_r2")),
        "mean_window_r2": _float_or_dash(metrics.get("mean_window_r2")),
        "frac_window_r2_positive": _float_or_dash(metrics.get("frac_window_r2_positive")),
    }

    if reference is not None and reference.run_name != bundle.run_name:
        ref_metrics = reference.metrics
        ref_baseline = ref_metrics.get("per_neuron_mean_nll_per_masked_bin")
        ref_model = ref_metrics.get("model_nll_per_masked_bin")
        ref_nll_gain = None
        if ref_baseline is not None and ref_model is not None:
            ref_nll_gain = float(ref_baseline) - float(ref_model)

        for key, current, ref_value in (
            ("delta_bps_vs_reference", metrics.get("bits_per_spike_vs_per_neuron_mean"), ref_metrics.get("bits_per_spike_vs_per_neuron_mean")),
            ("delta_median_r2_vs_reference", metrics.get("median_window_r2"), ref_metrics.get("median_window_r2")),
            ("delta_nll_gain_vs_reference", nll_gain, ref_nll_gain),
            ("delta_corr_vs_reference", metrics.get("corr_masked_bins"), ref_metrics.get("corr_masked_bins")),
        ):
            if current is None or ref_value is None:
                row[key] = "-"
            else:
                row[key] = _float_or_dash(float(current) - float(ref_value))

    return row


def _write_summary_csv(bundles: Sequence[RunBundle], output_path: Path) -> None:
    rows = [_bundle_summary_row(bundle, reference=bundles[0]) for bundle in bundles]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_notes(notes: Sequence[str], output_path: Path) -> None:
    if not notes:
        return
    output_path.write_text("\n".join(f"- {note}" for note in notes) + "\n")


def _all_unique_runs(comparisons: Sequence[ComparisonSpec]) -> Dict[Tuple[str, str, str], RunDef]:
    unique: Dict[Tuple[str, str, str], RunDef] = {}
    for spec in comparisons:
        for eval_mode in spec.eval_modes:
            for run_def in spec.runs:
                unique.setdefault((run_def.run_name, spec.source_type, eval_mode), run_def)
    return unique


def _render_individual_runs(results_dir: Path, output_dir: Path, comparisons: Sequence[ComparisonSpec], plt) -> None:
    individual_root = output_dir / "individual_runs"
    individual_root.mkdir(parents=True, exist_ok=True)

    for (run_name, source_type, eval_mode), run_def in _all_unique_runs(comparisons).items():
        print(f"[INDIVIDUAL] {run_name} | {eval_mode}", flush=True)
        bundle = _prepare_bundle(results_dir, run_def, source_type, eval_mode)
        run_root = individual_root / _safe_name(run_name)
        run_root.mkdir(parents=True, exist_ok=True)

        suffix = eval_mode if source_type == "combined" else "evaluation"
        _plot_individual_histograms(bundle, run_root / f"histograms_{suffix}.png", plt)
        _plot_individual_time_lines(bundle, run_root / f"per_time_lines_{suffix}.png", plt)
        _write_summary_csv([bundle], run_root / f"summary_{suffix}.csv")


def _render_comparisons(results_dir: Path, output_dir: Path, comparisons: Sequence[ComparisonSpec], plt) -> pd.DataFrame:
    comparison_root = output_dir / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    for spec in comparisons:
        spec_root = comparison_root / spec.key
        spec_root.mkdir(parents=True, exist_ok=True)
        _write_notes(spec.notes, spec_root / "notes.txt")

        for eval_mode in spec.eval_modes:
            print(f"[COMPARISON] {spec.key} | {eval_mode}", flush=True)
            mode_root = spec_root / (f"{eval_mode}_evaluation" if spec.source_type == "combined" else eval_mode)
            mode_root.mkdir(parents=True, exist_ok=True)

            bundles = [_prepare_bundle(results_dir, run_def, spec.source_type, eval_mode) for run_def in spec.runs]
            _write_summary_csv(bundles, mode_root / "summary.csv")
            _plot_comparison_per_neuron_lines(spec, eval_mode, bundles, mode_root / "per_neuron_lines.png", plt)
            _plot_comparison_per_neuron_deltas(spec, eval_mode, bundles, mode_root / "per_neuron_deltas.png", plt)
            _plot_comparison_per_time_lines(spec, eval_mode, bundles, mode_root / "per_time_lines.png", plt)

            manifest_rows.append(
                {
                    "comparison_key": spec.key,
                    "title": spec.title,
                    "source_type": spec.source_type,
                    "eval_mode": eval_mode,
                    "run_count": len(spec.runs),
                    "runs": " | ".join(run.run_name for run in spec.runs),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "comparison_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_20ms"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_20ms/report/section1_thesis_plots"))
    parser.add_argument("--comparisons", nargs="*", default=None,
                        help="Optional subset of comparison keys to render")
    args = parser.parse_args()

    selected = COMPARISONS
    if args.comparisons:
        wanted = set(args.comparisons)
        selected = tuple(spec for spec in COMPARISONS if spec.key in wanted)
        missing = wanted - {spec.key for spec in selected}
        if missing:
            raise SystemExit(f"Unknown comparison keys: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = args.output_dir / "mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })

    _render_individual_runs(args.results_dir, args.output_dir, selected, plt)
    manifest = _render_comparisons(args.results_dir, args.output_dir, selected, plt)
    print(f"[INFO] rendered {len(manifest)} comparison mode folders")
    print(f"[INFO] outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
