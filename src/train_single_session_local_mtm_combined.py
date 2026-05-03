"""
train_single_session_local_mtm_combined.py

This script implements a direct IBL-MtM fine-tuning run on 
one local session, alternating between neuron and causal 
masking objectives per batch, and introduces the prompt token so that
the model knows which masking mode is active. 

Evaluation is performed separately for neuron and causal masking to 
track performance on both objectives.
"""

import csv
import json
import os
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from local_eval_artifacts import (
    compute_train_baselines,
    evaluate_and_save_artifacts,
    save_history_artifacts,
    save_pretrained_load_report,
    save_run_metadata,
)

from train_single_session_local import (
    LocalSessionDataset,
    cfg_set,
    collate_local_batch,
    load_pretrained_weights,
    make_model,
    make_optimizer,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
)
from utils.config_utils import config_from_kwargs, update_config


def build_config(args):
    """
    Builds the direct IBL-MtM fine-tuning config for one local session.

    The transfer modes:
    - stitching on
    - prompt token on
    - session token off locally
    - training alternates only between neuron and causal masking
    """
    kwargs = {
        "model": "include:src/configs/ndt1_stitching_prompting.yaml"
    }
    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/ndt1_stitching_prompting.yaml", config)
    config = update_config("src/configs/ssl_session_trainer.yaml", config)

    cfg_set(config, "seed", args.seed)
    cfg_set(config, "wandb.use", False)

    cfg_set(config, "wandb.project", "ibl-mtm-direct-finetune-combined")
    cfg_set(config, "training.num_epochs", args.epochs)
    cfg_set(config, "training.train_batch_size", args.batch_size)
    cfg_set(config, "training.test_batch_size", args.batch_size)

    cfg_set(config, "data.dataset_name", "local_ssl")
    cfg_set(config, "data.max_time_length", args.time_bins)
    cfg_set(config, "data.max_space_length", args.num_neurons)
    cfg_set(config, "data.load_meta", True)
    cfg_set(config, "data.spike_augmentation", False)
    cfg_set(config, "data.target", None)

    cfg_set(config, "model.model_class", "NDT1")
    cfg_set(config, "model.encoder.stitching", True)

    cfg_set(config, "model.encoder.masker.force_active", True)
    cfg_set(config, "model.encoder.masker.mode", "neuron")
    cfg_set(config, "model.encoder.masker.ratio", args.neuron_mask_ratio)
    cfg_set(config, "model.encoder.masker.zero_ratio", 1.0)
    cfg_set(config, "model.encoder.masker.random_ratio", 0.0)
    cfg_set(config, "model.encoder.masker.expand_prob", 0.0)
    cfg_set(config, "model.encoder.masker.max_timespan", 1)
    cfg_set(config, "model.encoder.masker.mask_regions", ["all"])
    cfg_set(config, "model.encoder.masker.target_regions", ["all"])
    cfg_set(config, "model.encoder.masker.n_mask_regions", 1)
    cfg_set(config, "model.encoder.embedder.max_F", args.time_bins)
    cfg_set(config, "model.encoder.embedder.use_prompt", True)

    cfg_set(config, "model.encoder.embedder.use_session", True)

    # Causal masking is implemented by setting the backward context to -1 and 
    # forward context to 0, so that only past bins are visible 
    # for prediction. 
    cfg_set(config, "model.encoder.context.forward", 0)
    cfg_set(config, "model.encoder.context.backward", -1)

    cfg_set(config, "optimizer.lr", args.lr)
    cfg_set(config, "optimizer.wd", args.weight_decay)
    cfg_set(config, "optimizer.warmup_pct", args.warmup_pct)
    cfg_set(config, "optimizer.eps", 1e-8)
    return config


def _validate_direct_transfer_setup(config, load_report: dict, num_neurons: int) -> None:
        """ 
    Validates that the pretrained checkpoint load report 
    is consistent with a clean direct IBL-MtM fine-tuning setup,
        """
        
    if not bool(config.model.encoder.stitching):
        raise RuntimeError("Combined script is misconfigured: stitching must stay enabled for direct IBL-MtM fine-tuning.")
    if not bool(config.model.encoder.embedder.use_prompt):
        raise RuntimeError("Combined script is misconfigured: prompt token must stay enabled for direct IBL-MtM fine-tuning.")
    if not bool(config.model.encoder.embedder.use_session):
        raise RuntimeError("Combined script is misconfigured: use_session must be enabled to match IBL-MtM fine-tuning.")

    if int(load_report.get("loaded_tensor_count", 0)) <= 0:
        raise RuntimeError("Pretrained checkpoint load reused zero tensors. That is not direct IBL-MtM fine-tuning.")
    if int(load_report.get("unexpected_tensor_count", 0)) != 0:
        raise RuntimeError(f"Checkpoint load produced unexpected tensors: {load_report.get('unexpected_keys', [])}")

    allowed_missing_prefixes = (
        f"encoder.stitcher.stitcher_dict.{num_neurons}.",
        f"stitch_decoder.stitch_decoder_dict.{num_neurons}.",
        # embed_session is shape-mismatched in the pretrained checkpoint (different
        # number of sessions), so it is skipped and randomly initialized locally.
        "encoder.embedder.embed_session.",
    )
    unexpected_missing = [
        key for key in load_report.get("missing_keys", [])
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected_missing:
        raise RuntimeError(
            "Checkpoint load is missing non-local-transfer tensors, so this is not a clean direct IBL-MtM fine-tune: "
            f"{unexpected_missing}"
        )


def _safe_float_div(total_loss: float, total_examples: int) -> float:
    return total_loss / max(total_examples, 1)


def _make_epoch_schedule(num_steps: int, causal_prob: float) -> List[str]:
    """ 
    Creates a randomized schedule of masking modes for 
    one training epoch.
    """
    if num_steps <= 0:
        return []
    if num_steps == 1:
        return ["causal" if causal_prob >= 0.5 else "neuron"]

    n_causal = int(round(causal_prob * num_steps))
    if causal_prob > 0.0:
        n_causal = max(1, n_causal)
    if causal_prob < 1.0:
        n_causal = min(num_steps - 1, n_causal)
    schedule = ["causal"] * n_causal + ["neuron"] * (num_steps - n_causal)
    random.shuffle(schedule)
    return schedule


def _forward_outputs(model, batch, masking_mode: str):
    """ 
    Computes forward pass and returns outputs for one batch.
    """
    return model(
        batch["spikes_data"],
        time_attn_mask=batch["time_attn_mask"],
        space_attn_mask=batch["space_attn_mask"],
        spikes_timestamps=batch["spikes_timestamps"],
        spikes_spacestamps=batch["spikes_spacestamps"],
        targets=batch["target"],
        neuron_regions=batch["neuron_regions"],
        masking_mode=masking_mode,
        spike_augmentation=False,
        num_neuron=batch["spikes_data"].shape[2],
        eid=batch["eid"][0],
    )


def run_train_epoch_mixed(model, loader, optimizer, device, masking_ratios: Dict[str, float], causal_prob: float, max_batches: int = None, lr_scheduler=None):
    """
    Runs one training epoch,
    alternating between neuron and causal masking per batch.
    """
    model.train()
    total_loss = 0.0
    total_examples = 0
    mode_totals = {
        "neuron": {"loss": 0.0, "examples": 0, "batches": 0},
        "causal": {"loss": 0.0, "examples": 0, "batches": 0},
    }

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    mode_schedule = _make_epoch_schedule(num_steps, causal_prob)

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        masking_mode = mode_schedule[step]
        model.encoder.masker.ratio = masking_ratios[masking_mode]
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_outputs(model, batch, masking_mode)
        loss = outputs.loss
        n_examples = int(outputs.n_examples.item()) if torch.is_tensor(outputs.n_examples) else int(outputs.n_examples)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += float(loss.item())
        total_examples += max(n_examples, 1)
        mode_totals[masking_mode]["loss"] += float(loss.item())
        mode_totals[masking_mode]["examples"] += max(n_examples, 1)
        mode_totals[masking_mode]["batches"] += 1

    return {
        "train_loss_per_masked_bin": _safe_float_div(total_loss, total_examples),
        "train_neuron_loss_per_masked_bin": _safe_float_div(mode_totals["neuron"]["loss"], mode_totals["neuron"]["examples"]) if mode_totals["neuron"]["batches"] else float("nan"),
        "train_causal_loss_per_masked_bin": _safe_float_div(mode_totals["causal"]["loss"], mode_totals["causal"]["examples"]) if mode_totals["causal"]["batches"] else float("nan"),
        "train_neuron_batches": mode_totals["neuron"]["batches"],
        "train_causal_batches": mode_totals["causal"]["batches"],
    }


def run_eval_epoch_per_mode(model, loader, device, masking_ratios: Dict[str, float], max_batches: int = None):
    model.eval()
    results = {}
    with torch.no_grad():
        for masking_mode in ("neuron", "causal"):
            model.encoder.masker.ratio = masking_ratios[masking_mode]
            total_loss = 0.0
            total_examples = 0

            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                outputs = _forward_outputs(model, batch, masking_mode)
                loss = outputs.loss
                n_examples = int(outputs.n_examples.item()) if torch.is_tensor(outputs.n_examples) else int(outputs.n_examples)
                total_loss += float(loss.item())
                total_examples += max(n_examples, 1)

            results[masking_mode] = _safe_float_div(total_loss, total_examples)

    results["mean"] = float((results["neuron"] + results["causal"]) / 2.0)
    return results


def save_combined_eval_summary(save_dir: str, summary_by_mode: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Saves a combined evaluation summary for all masking modes and splits,
    """
    artifact_dir = Path(save_dir) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "combined_eval_summary.json", "w") as f:
        json.dump(summary_by_mode, f, indent=2)

    rows = []
    for masking_mode, split_summaries in summary_by_mode.items():
        for split_name, metrics in split_summaries.items():
            row = {"masking_mode": masking_mode, "split": split_name}
            row.update(metrics)
            rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(artifact_dir / "combined_eval_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    gap_payload = {}
    for masking_mode, split_summaries in summary_by_mode.items():
        if "train" in split_summaries and "test" in split_summaries:
            train_metrics = split_summaries["train"]
            test_metrics = split_summaries["test"]
            gap_payload[masking_mode] = {
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

    if gap_payload:
        mean_gap = {
            "test_minus_train_model_nll_per_masked_bin_mean": float(sum(v["test_minus_train_model_nll_per_masked_bin"] for v in gap_payload.values()) / len(gap_payload)),
            "test_minus_train_bits_per_spike_vs_per_neuron_mean_mean": float(sum(v["test_minus_train_bits_per_spike_vs_per_neuron_mean"] for v in gap_payload.values()) / len(gap_payload)),
        }
        gap_payload["mean_over_modes"] = mean_gap
        with open(artifact_dir / "combined_generalization_gap.json", "w") as f:
            json.dump(gap_payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Absolute path to the preprocessed control-session directory")
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="Path to pretrained IBL MtM checkpoint (.pt)")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="LR for transferred/shared pretrained parameters")
    parser.add_argument("--adapter-lr", type=float, default=None,
                        help="Optional LR for newly initialized local stitcher/decoder parameters; omit for a plain single-LR run")
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--warmup-pct", type=float, default=0.2,
                        help="OneCycleLR warmup fraction (pct_start); saved into metadata and used directly by the scheduler")
    parser.add_argument("--neuron-mask-ratio", type=float, default=0.3)
    parser.add_argument("--causal-mask-ratio", type=float, default=0.6)
    parser.add_argument("--causal-prob", type=float, default=0.5,
                        help="Fraction of training batches assigned to causal masking; remainder use neuron masking")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stop after this many epochs without a meaningful mean val improvement")
    parser.add_argument("--min-delta", type=float, default=1e-5,
                        help="Minimum mean val-loss improvement required to reset early-stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Runs only a few batches for train/val")
    parser.add_argument("--eval-splits", nargs="*", default=["test", "train", "val"],
                        choices=["train", "val", "test"],
                        help="Splits to evaluate after training; order controls which subdirectories are produced")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for artifact evaluation; defaults to --batch-size")
    parser.add_argument("--eval-neuron-mask-ratio", type=float, default=None,
                        help="Held-out neuron ratio for artifact evaluation; defaults to --neuron-mask-ratio")
    parser.add_argument("--eval-causal-mask-ratio", type=float, default=None,
                        help="Held-out causal ratio for artifact evaluation; defaults to --causal-mask-ratio")
    parser.add_argument("--artifact-max-batches", type=int, default=None,
                        help="Optional cap for quick artifact generation during smoke jobs")
    parser.add_argument("--artifact-seed", type=int, default=123,
                        help="Seed for deterministic held-out artifact masks")
    parser.add_argument("--skip-artifacts", action="store_true",
                        help="Skip post-training prediction tables and plots")
    parser.add_argument("--skip-latent-artifacts", action="store_true",
                        help="Skip latent PCA artifacts while keeping prediction artifacts")
    args = parser.parse_args()

    set_seed(args.seed)

    train_ds = LocalSessionDataset(args.data_dir, split="train")
    val_ds = LocalSessionDataset(args.data_dir, split="val")
    test_ds = LocalSessionDataset(args.data_dir, split="test")
    args.time_bins = train_ds.T
    args.num_neurons = train_ds.N


    print("[BIG RUN] direct IBL-MtM combined fine-tuning")
    print(f"[INFO] train windows: {len(train_ds)}")
    print(f"[INFO] val windows:   {len(val_ds)}")
    print(f"[INFO] test windows:  {len(test_ds)}")
    print(f"[INFO] shape per example: [{train_ds.T}, {train_ds.N}]")

    config = build_config(args)
    print("[CONFIG]")
    print(f"  pretrained_ckpt     = {args.pretrained_ckpt}")
    print(f"  masking_modes       = ['neuron', 'causal']")
    print(f"  neuron_mask_ratio   = {args.neuron_mask_ratio}")
    print(f"  causal_mask_ratio   = {args.causal_mask_ratio}")
    print(f"  causal_prob         = {args.causal_prob}")
    print(f"  stitching           = {config.model.encoder.stitching}")
    print(f"  use_prompt          = {config.model.encoder.embedder.use_prompt}")
    print(f"  use_session         = {config.model.encoder.embedder.use_session}")
    print(f"  batch_size          = {args.batch_size}")
    print(f"  epochs              = {args.epochs}")
    print(f"  lr                  = {args.lr}")
    print(f"  adapter_lr          = {args.adapter_lr}")
    print(f"  weight_decay        = {args.weight_decay}")
    print(f"  warmup_pct          = {args.warmup_pct}")
    print(f"  patience            = {args.patience}")
    print(f"  min_delta           = {args.min_delta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = make_model(config, num_neurons=train_ds.N).to(device)
    load_report = load_pretrained_weights(model, args.pretrained_ckpt, device)
    # Abort before training if transfer did not happen cleanly.
    _validate_direct_transfer_setup(config, load_report, train_ds.N)
    optimizer = make_optimizer(
        model=model,
        num_neurons=train_ds.N,
        base_lr=config.optimizer.lr,
        adapter_lr=args.adapter_lr,
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_local_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_local_batch,
    )

    # OneCycleLR stepped per batch,
    steps_per_epoch = 3 if args.smoke_test else len(train_loader)
    max_lr = [config.optimizer.lr, args.adapter_lr] if args.adapter_lr is not None else config.optimizer.lr
    lr_scheduler = OneCycleLR(
        optimizer,
        total_steps=args.epochs * steps_per_epoch,
        max_lr=max_lr,
        pct_start=args.warmup_pct,
        div_factor=10,
    )
    print(
        f"[INFO] OneCycleLR: total_steps={args.epochs * steps_per_epoch}, "
        f"max_lr={max_lr}, pct_start={args.warmup_pct}, div_factor=10"
    )


    first_batch = next(iter(train_loader))
    print("[INFO] first batch shapes:")
    print("  spikes_data        ", tuple(first_batch["spikes_data"].shape))
    print("  time_attn_mask     ", tuple(first_batch["time_attn_mask"].shape))
    print("  space_attn_mask    ", tuple(first_batch["space_attn_mask"].shape))
    print("  spikes_timestamps  ", tuple(first_batch["spikes_timestamps"].shape))
    print("  spikes_spacestamps ", tuple(first_batch["spikes_spacestamps"].shape))
    print("  neuron_regions     ", first_batch["neuron_regions"].shape)
    print("  eid[0]             ", first_batch["eid"][0])

    batch = move_batch_to_device(first_batch, device)
    masking_ratios = {"neuron": args.neuron_mask_ratio, "causal": args.causal_mask_ratio}
    model.eval()
    with torch.no_grad():
        for sanity_mode in ("neuron", "causal"):
            model.encoder.masker.ratio = masking_ratios[sanity_mode]
            sanity_outputs = _forward_outputs(model, batch, sanity_mode)
            print(f"[INFO] sanity {sanity_mode} loss: {sanity_outputs.loss.item():.6f}")
            print(f"[INFO] sanity {sanity_mode} n_examples: {int(sanity_outputs.n_examples.item())}")

    max_batches = 3 if args.smoke_test else None
    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    save_run_metadata(
        args.save_dir,
        args,
        config,
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )
    save_pretrained_load_report(args.save_dir, load_report)

    history = []
    training_start_time = time.perf_counter()
    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        train_stats = run_train_epoch_mixed(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            masking_ratios=masking_ratios,
            causal_prob=args.causal_prob,
            max_batches=max_batches,
            lr_scheduler=lr_scheduler,
        )
        val_stats = run_eval_epoch_per_mode(
            model=model,
            loader=val_loader,
            device=device,
            masking_ratios=masking_ratios,
            max_batches=max_batches,
        )

        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - training_start_time
        mean_epoch_seconds = elapsed_seconds / max(epoch + 1, 1)
        remaining_epochs = max(args.epochs - epoch - 1, 0)
        eta_seconds = remaining_epochs * mean_epoch_seconds

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_mixed={train_stats['train_loss_per_masked_bin']:.8f} "
            f"train_neuron={train_stats['train_neuron_loss_per_masked_bin']:.8f} "
            f"train_causal={train_stats['train_causal_loss_per_masked_bin']:.8f} "
            f"val_mean={val_stats['mean']:.8f} "
            f"val_neuron={val_stats['neuron']:.8f} "
            f"val_causal={val_stats['causal']:.8f} "
            f"epoch_seconds={epoch_seconds:.1f} "
            f"eta_hours={eta_seconds / 3600:.2f}"
        )

        improved = val_stats["mean"] < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_stats["mean"]
            epochs_since_improvement = 0
            save_checkpoint(args.save_dir, "best", model, optimizer, epoch, best_val_loss)
            print(f"[INFO] new best mean val loss: {best_val_loss:.8f}")
        else:
            epochs_since_improvement += 1

        save_checkpoint(args.save_dir, "last", model, optimizer, epoch, best_val_loss)

        history.append({
            "epoch": epoch,
            "train_loss_per_masked_bin": train_stats["train_loss_per_masked_bin"],
            "val_loss_per_masked_bin": val_stats["mean"],
            "best_val_loss_so_far": best_val_loss,
            "train_neuron_loss_per_masked_bin": train_stats["train_neuron_loss_per_masked_bin"],
            "train_causal_loss_per_masked_bin": train_stats["train_causal_loss_per_masked_bin"],
            "train_neuron_batches": train_stats["train_neuron_batches"],
            "train_causal_batches": train_stats["train_causal_batches"],
            "val_neuron_loss_per_masked_bin": val_stats["neuron"],
            "val_causal_loss_per_masked_bin": val_stats["causal"],
        })
        save_history_artifacts(args.save_dir, history)

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(
                f"[INFO] early stopping at epoch {epoch:03d}: "
                f"no mean val improvement >= {args.min_delta:g} for {args.patience} epochs"
            )
            break

    print(f"[DONE] Best mean val loss: {best_val_loss:.8f}")
    print(f"[DONE] Checkpoints saved to: {args.save_dir}")

    if not args.skip_artifacts:
        best_ckpt_path = os.path.join(args.save_dir, "best.pt")
        print(f"[INFO] loading best checkpoint for artifacts: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        baselines = compute_train_baselines(train_ds)
        split_to_dataset = {"train": train_ds, "val": val_ds, "test": test_ds}
        summary_by_mode = {"neuron": {}, "causal": {}}
        eval_mask_ratios = {
            "neuron": args.eval_neuron_mask_ratio if args.eval_neuron_mask_ratio is not None else args.neuron_mask_ratio,
            "causal": args.eval_causal_mask_ratio if args.eval_causal_mask_ratio is not None else args.causal_mask_ratio,
        }

        for split_name in args.eval_splits:
            eval_ds = split_to_dataset[split_name]
            eval_loader = DataLoader(
                eval_ds,
                batch_size=args.eval_batch_size or args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_local_batch,
            )

            for masking_mode in ("neuron", "causal"):
                artifact_subdir = f"eval_{split_name}_{masking_mode}"
                # Save latents for every requested split so latent-space
                # generalisation can test whether train/val/test windows intermix.
                save_latents = not args.skip_latent_artifacts
                summary_by_mode[masking_mode][split_name] = evaluate_and_save_artifacts(
                    model=model,
                    eval_loader=eval_loader,
                    device=device,
                    save_dir=args.save_dir,
                    masking_mode=masking_mode,
                    eval_split=split_name,
                    mask_ratio=eval_mask_ratios[masking_mode],
                    baselines=baselines,
                    use_lograte=bool(config.method.model_kwargs.use_lograte),
                    eval_seed=args.artifact_seed,
                    max_batches=args.artifact_max_batches,
                    save_latents=save_latents,
                    artifact_subdir=artifact_subdir,
                )

        save_combined_eval_summary(args.save_dir, summary_by_mode)


if __name__ == "__main__":
    main()
