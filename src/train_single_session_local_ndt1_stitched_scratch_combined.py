
"""
train_single_session_local_ndt1_stitched_scratch_combined.py

Train a single-session NDT1 from scratch with both neuron 
and causal objectives combined.
"""
import os
import argparse
import random
import time
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
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
    move_batch_to_device,
    save_checkpoint,
    set_seed,
)
from train_single_session_local_mtm_combined import save_combined_eval_summary


def build_config(args):
    """
    Stitched NDT1 from scratch with combined (neuron + causal) objective.

    Direct ablation of
    pretraining: same objective, same architecture, random init.

    Attention is fixed to causal (matches mtm_combined).  
    The masker mode is overridden per batch at runtime; 
    the config default is just a placeholder.
    """
    kwargs = {"model": "include:src/configs/ndt1_stitching_prompting.yaml"}
    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/ndt1_stitching_prompting.yaml", config)
    config = update_config("src/configs/ssl_session_trainer.yaml", config)

    cfg_set(config, "seed", args.seed)
    cfg_set(config, "wandb.use", False)
    cfg_set(config, "wandb.project", "single-session-local-ndt1-stitched-scratch-combined")
    cfg_set(config, "training.num_epochs", args.epochs)
    cfg_set(config, "training.train_batch_size", args.batch_size)
    cfg_set(config, "training.test_batch_size", args.batch_size)

    cfg_set(config, "data.dataset_name", "local_ssl")
    cfg_set(config, "data.max_time_length", args.time_bins)
    cfg_set(config, "data.max_space_length", args.num_neurons)
    cfg_set(config, "data.load_meta", True)
    cfg_set(config, "data.spike_augmentation", False)
    cfg_set(config, "data.target", None)

    # Same architecture as the IBL-MtM fine-tune runs.
    cfg_set(config, "model.model_class", "NDT1")
    cfg_set(config, "model.encoder.stitching", True)
    cfg_set(config, "model.encoder.embedder.max_F", args.time_bins)
    cfg_set(config, "model.encoder.embedder.use_prompt", True)

    # embedding is created from data/target_eids.txt and learned from scratch.
    cfg_set(config, "model.encoder.embedder.use_session", True)

    # Causal attention context (matching mtm_combined).
    cfg_set(config, "model.encoder.context.forward", 0)
    cfg_set(config, "model.encoder.context.backward", -1)

    # Masker — mode overridden per batch; ratio set at runtime.
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

    cfg_set(config, "optimizer.lr", args.lr)
    cfg_set(config, "optimizer.wd", args.weight_decay)
    cfg_set(config, "optimizer.eps", 1e-8)
    return config


def make_model(config, num_neurons: int):
    return NDT1(
        config.model,
        **config.method.model_kwargs,
        num_neurons=[num_neurons],
    )


def _make_epoch_schedule(num_steps: int, causal_prob: float) -> List[str]:
    """Balanced shuffled schedule so both modes appear every epoch."""
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


def _forward(model, batch, masking_mode: str):
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


def run_train_epoch_mixed(model, loader, optimizer, device, masking_ratios: Dict[str, float],
                          causal_prob: float, max_batches: int = None, lr_scheduler=None):
    model.train()
    total_loss, total_examples = 0.0, 0
    mode_totals = {m: {"loss": 0.0, "examples": 0, "batches": 0} for m in ("neuron", "causal")}

    num_steps = len(loader) if max_batches is None else min(len(loader), max_batches)
    schedule = _make_epoch_schedule(num_steps, causal_prob)

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break
        mode = schedule[step]
        model.encoder.masker.ratio = masking_ratios[mode]
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch, mode)
        loss = outputs.loss
        n = int(outputs.n_examples.item()) if torch.is_tensor(outputs.n_examples) else int(outputs.n_examples)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        total_loss += float(loss.item())
        total_examples += max(n, 1)
        mode_totals[mode]["loss"] += float(loss.item())
        mode_totals[mode]["examples"] += max(n, 1)
        mode_totals[mode]["batches"] += 1

    def _avg(m):
        return mode_totals[m]["loss"] / max(mode_totals[m]["examples"], 1) if mode_totals[m]["batches"] else float("nan")

    return {
        "train_loss_per_masked_bin":         total_loss / max(total_examples, 1),
        "train_neuron_loss_per_masked_bin":  _avg("neuron"),
        "train_causal_loss_per_masked_bin":  _avg("causal"),
        "train_neuron_batches":              mode_totals["neuron"]["batches"],
        "train_causal_batches":              mode_totals["causal"]["batches"],
    }


def run_eval_epoch_per_mode(model, loader, device, masking_ratios: Dict[str, float], max_batches: int = None):
    model.eval()
    results = {}
    with torch.no_grad():
        for mode in ("neuron", "causal"):
            model.encoder.masker.ratio = masking_ratios[mode]
            total_loss, total_examples = 0.0, 0
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                outputs = _forward(model, batch, mode)
                n = int(outputs.n_examples.item()) if torch.is_tensor(outputs.n_examples) else int(outputs.n_examples)
                total_loss += float(outputs.loss.item())
                total_examples += max(n, 1)
            results[mode] = total_loss / max(total_examples, 1)
    results["mean"] = float((results["neuron"] + results["causal"]) / 2.0)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--neuron-mask-ratio", type=float, default=0.3)
    parser.add_argument("--causal-mask-ratio", type=float, default=0.6)
    parser.add_argument("--causal-prob", type=float, default=0.5,
                        help="Fraction of training batches assigned to causal masking")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--eval-splits", nargs="*", default=["test", "train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-neuron-mask-ratio", type=float, default=None)
    parser.add_argument("--eval-causal-mask-ratio", type=float, default=None)
    parser.add_argument("--artifact-max-batches", type=int, default=None)
    parser.add_argument("--artifact-seed", type=int, default=123)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--skip-latent-artifacts", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    train_ds = LocalSessionDataset(args.data_dir, split="train")
    val_ds   = LocalSessionDataset(args.data_dir, split="val")
    test_ds  = LocalSessionDataset(args.data_dir, split="test")
    args.time_bins   = train_ds.T
    args.num_neurons = train_ds.N

    print("[BASELINE] stitched NDT1 scratch — combined neuron+causal objective")
    print(f"[INFO] train windows: {len(train_ds)}")
    print(f"[INFO] val windows:   {len(val_ds)}")
    print(f"[INFO] test windows:  {len(test_ds)}")
    print(f"[INFO] shape per example: [{train_ds.T}, {train_ds.N}]")

    config = build_config(args)
    print("[CONFIG]")
    print(f"  masking_modes       = ['neuron', 'causal']")
    print(f"  neuron_mask_ratio   = {args.neuron_mask_ratio}")
    print(f"  causal_mask_ratio   = {args.causal_mask_ratio}")
    print(f"  causal_prob         = {args.causal_prob}")
    print(f"  stitching           = {config.model.encoder.stitching}")
    print(f"  use_prompt          = {config.model.encoder.embedder.use_prompt}")
    print(f"  use_session         = {config.model.encoder.embedder.use_session}")
    print(f"  lr                  = {args.lr}")
    print(f"  patience            = {args.patience}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = make_model(config, num_neurons=train_ds.N).to(device)
    load_report = {
        "mode": "scratch_stitched_ndt1_combined",
        "checkpoint_path": None,
        "loaded_tensor_count": 0,
        "skipped_tensor_count": 0,
        "missing_tensor_count": 0,
        "unexpected_tensor_count": 0,
        "skipped_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
    }

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=collate_local_batch)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=collate_local_batch)

    steps_per_epoch = 3 if args.smoke_test else len(train_loader)
    lr_scheduler = OneCycleLR(optimizer, total_steps=args.epochs * steps_per_epoch,
                               max_lr=args.lr, pct_start=0.15, div_factor=10)
    print(f"[INFO] OneCycleLR: total_steps={args.epochs * steps_per_epoch}, max_lr={args.lr}")

    masking_ratios = {"neuron": args.neuron_mask_ratio, "causal": args.causal_mask_ratio}

    first_batch = next(iter(train_loader))
    batch = move_batch_to_device(first_batch, device)
    model.eval()
    with torch.no_grad():
        for mode in ("neuron", "causal"):
            model.encoder.masker.ratio = masking_ratios[mode]
            out = _forward(model, batch, mode)
            print(f"[INFO] sanity {mode} loss: {out.loss.item():.6f}")

    max_batches = 3 if args.smoke_test else None
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    os.makedirs(args.save_dir, exist_ok=True)
    save_run_metadata(args.save_dir, args, config,
                      {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)})
    save_pretrained_load_report(args.save_dir, load_report)

    history = []
    training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_stats = run_train_epoch_mixed(model, train_loader, optimizer, device,
                                            masking_ratios, args.causal_prob,
                                            max_batches, lr_scheduler)
        val_stats = run_eval_epoch_per_mode(model, val_loader, device, masking_ratios, max_batches)

        epoch_s = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - training_start_time
        eta_s   = (elapsed / max(epoch + 1, 1)) * max(args.epochs - epoch - 1, 0)
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_mixed={train_stats['train_loss_per_masked_bin']:.8f} "
            f"train_neuron={train_stats['train_neuron_loss_per_masked_bin']:.8f} "
            f"train_causal={train_stats['train_causal_loss_per_masked_bin']:.8f} "
            f"val_mean={val_stats['mean']:.8f} "
            f"val_neuron={val_stats['neuron']:.8f} "
            f"val_causal={val_stats['causal']:.8f} "
            f"epoch_s={epoch_s:.1f} eta_h={eta_s/3600:.2f}"
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
            "train_loss_per_masked_bin":        train_stats["train_loss_per_masked_bin"],
            "train_neuron_loss_per_masked_bin":  train_stats["train_neuron_loss_per_masked_bin"],
            "train_causal_loss_per_masked_bin":  train_stats["train_causal_loss_per_masked_bin"],
            "val_loss_per_masked_bin":           val_stats["mean"],
            "val_neuron_loss_per_masked_bin":    val_stats["neuron"],
            "val_causal_loss_per_masked_bin":    val_stats["causal"],
            "best_val_loss_so_far":              best_val_loss,
        })
        save_history_artifacts(args.save_dir, history)

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(f"[INFO] early stopping at epoch {epoch:03d}: "
                  f"no mean val improvement >= {args.min_delta:g} for {args.patience} epochs")
            break

    print(f"[DONE] Best mean val loss: {best_val_loss:.8f}")

    if not args.skip_artifacts:
        best_ckpt = torch.load(os.path.join(args.save_dir, "best.pt"),
                               map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        baselines = compute_train_baselines(train_ds)
        split_to_ds = {"train": train_ds, "val": val_ds, "test": test_ds}
        eval_mask_ratios = {
            "neuron": args.eval_neuron_mask_ratio if args.eval_neuron_mask_ratio is not None else args.neuron_mask_ratio,
            "causal": args.eval_causal_mask_ratio if args.eval_causal_mask_ratio is not None else args.causal_mask_ratio,
        }
        summary_by_mode = {"neuron": {}, "causal": {}}

        for split_name in args.eval_splits:
            eval_ds = split_to_ds[split_name]
            eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size or args.batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=torch.cuda.is_available(),
                                     collate_fn=collate_local_batch)
            for mode in ("neuron", "causal"):
                # Save latents for every requested split so latent-space
                # generalisation can test whether train/val/test windows intermix.
                save_latents = not args.skip_latent_artifacts
                summary_by_mode[mode][split_name] = evaluate_and_save_artifacts(
                    model=model,
                    eval_loader=eval_loader,
                    device=device,
                    save_dir=args.save_dir,
                    masking_mode=mode,
                    eval_split=split_name,
                    mask_ratio=eval_mask_ratios[mode],
                    baselines=baselines,
                    use_lograte=bool(config.method.model_kwargs.use_lograte),
                    eval_seed=args.artifact_seed,
                    max_batches=args.artifact_max_batches,
                    save_latents=save_latents,
                    artifact_subdir=f"eval_{split_name}_{mode}",
                )
        save_combined_eval_summary(args.save_dir, summary_by_mode)


if __name__ == "__main__":
    main()
