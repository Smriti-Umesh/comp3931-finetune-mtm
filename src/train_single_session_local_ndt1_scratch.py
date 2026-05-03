"""
train_single_session_local_ndt1_scratch.py
This script trains a single-session NDT1 model from scratch on
single-session data, without any pretrained checkpoints. 

NDT1 is trained from scratch as a baseline to compare
against the direct pretrained transfer and combined fine-tuning runs.

Handles neuron and causal masking depending on masking mode set in 

Example:
--masking-mode neuron
--masking-mode causal
"""

import os
import argparse
import time

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
    save_split_eval_summary,
)

from train_single_session_local import (
    LocalSessionDataset,
    cfg_set,
    collate_local_batch,
    move_batch_to_device,
    save_checkpoint,
    set_seed,
)


def build_config(args):
    """
    Builds a plain single-session NDT1 config trained from scratch.
    """
    model_config_path = "src/configs/ndt1_causal.yaml" if args.masking_mode == "causal" else "src/configs/ndt1.yaml"
    kwargs = {
        "model": f"include:{model_config_path}"
    }
    config = config_from_kwargs(kwargs)
    config = update_config(model_config_path, config)
    config = update_config("src/configs/ssl_session_trainer.yaml", config)

    # Keeping training metadata in the saved config so later comparisons can be made.
    cfg_set(config, "seed", args.seed)
    cfg_set(config, "wandb.use", False)
    cfg_set(config, "wandb.project", "single-session-local-ndt1-scratch")
    cfg_set(config, "training.num_epochs", args.epochs)
    cfg_set(config, "training.train_batch_size", args.batch_size)
    cfg_set(config, "training.test_batch_size", args.batch_size)

    cfg_set(config, "data.dataset_name", "local_ssl")
    cfg_set(config, "data.max_time_length", args.time_bins)
    cfg_set(config, "data.max_space_length", args.num_neurons)
    cfg_set(config, "data.load_meta", True)
    cfg_set(config, "data.spike_augmentation", False)
    cfg_set(config, "data.target", None)

    # Plain direct NDT1 baseline: no stitching, no prompt token, no session token.
    cfg_set(config, "model.model_class", "NDT1")
    cfg_set(config, "model.encoder.stitching", False)
    cfg_set(config, "model.encoder.embedder.n_channels", args.num_neurons)
    cfg_set(config, "model.encoder.embedder.max_F", args.time_bins)
    cfg_set(config, "model.encoder.embedder.use_prompt", False)
    cfg_set(config, "model.encoder.embedder.use_session", False)

    # Matching the evaluation objective with the chosen masking mode.
    cfg_set(config, "model.encoder.masker.force_active", True)
    cfg_set(config, "model.encoder.masker.ratio", args.mask_ratio)
    cfg_set(config, "model.encoder.masker.zero_ratio", 1.0)
    cfg_set(config, "model.encoder.masker.random_ratio", 0.0)
    cfg_set(config, "model.encoder.masker.expand_prob", 0.0)
    cfg_set(config, "model.encoder.masker.max_timespan", 1)
    cfg_set(config, "model.encoder.masker.mask_regions", ["all"])
    cfg_set(config, "model.encoder.masker.target_regions", ["all"])
    cfg_set(config, "model.encoder.masker.n_mask_regions", 1)

    if args.masking_mode == "causal":
        # Causal uses temporal masking plus a causal attention context.
        cfg_set(config, "model.encoder.masker.mode", "temporal")
        cfg_set(config, "model.encoder.context.forward", 0)
        cfg_set(config, "model.encoder.context.backward", -1)
    else:
        # Neuron masking lets the model use all timesteps while hiding selected neurons.
        cfg_set(config, "model.encoder.masker.mode", "neuron")
        cfg_set(config, "model.encoder.context.forward", -1)
        cfg_set(config, "model.encoder.context.backward", -1)

    cfg_set(config, "optimizer.lr", args.lr)
    cfg_set(config, "optimizer.wd", args.weight_decay)
    cfg_set(config, "optimizer.eps", 1e-8)
    return config


def make_model(config, num_neurons: int):
    # NDT1 still expects `num_neurons` in kwargs even when stitching is off.
    # or else leads to an error
    return NDT1(
        config.model,
        **config.method.model_kwargs,
        num_neurons=[num_neurons],
    )


def run_epoch(model, loader, optimizer, device, masking_mode: str, train: bool, max_batches: int = None, lr_scheduler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_examples = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(
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
            loss = outputs.loss
            n_examples = int(outputs.n_examples.item()) if torch.is_tensor(outputs.n_examples) else int(outputs.n_examples)
            if train:
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

        total_loss += float(loss.item())
        total_examples += max(n_examples, 1)

    return total_loss / max(total_examples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Absolute path to the preprocessed control-session directory")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--masking-mode", type=str, default="neuron", choices=["neuron", "causal"],
                        help="Baseline objective to train: held-out neurons or held-out future time bins")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Single LR for all trainable parameters in the direct NDT1 baseline")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.3,
                        help="For causal, set this to 0.6 if you want the same held-out horizon as the MtM causal runs")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop after this many epochs without a meaningful val improvement; 0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum val-loss improvement required to reset early-stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Runs only a few batches for train/val")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"],
                        help="Split used for post-training deterministic artifact evaluation")
    parser.add_argument("--extra-eval-splits", nargs="*", default=[],
                        choices=["train", "val", "test"],
                        help="Optional additional splits to evaluate and save under artifacts/eval_<split>")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for artifact evaluation; defaults to --batch-size")
    parser.add_argument("--eval-mask-ratio", type=float, default=None,
                        help="Held-out ratio for artifact evaluation; defaults to --mask-ratio")
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

    print("[BASELINE] direct single-session NDT1 scratch")
    print(f"[INFO] train windows: {len(train_ds)}")
    print(f"[INFO] val windows:   {len(val_ds)}")
    print(f"[INFO] test windows:  {len(test_ds)}")
    print(f"[INFO] shape per example: [{train_ds.T}, {train_ds.N}]")

    config = build_config(args)
    print("[CONFIG]")
    print(f"  masking_mode    = {args.masking_mode}")
    print(f"  mask_ratio      = {config.model.encoder.masker.ratio}")
    print(f"  stitching       = {config.model.encoder.stitching}")
    print(f"  use_prompt      = {config.model.encoder.embedder.use_prompt}")
    print(f"  use_session     = {config.model.encoder.embedder.use_session}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  epochs          = {args.epochs}")
    print(f"  lr              = {args.lr}")
    print(f"  patience        = {args.patience}")
    print(f"  min_delta       = {args.min_delta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = make_model(config, num_neurons=train_ds.N).to(device)

    load_report = {
        "mode": "scratch_direct_ndt1",
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
    print("[INFO] optimizer parameter groups:")
    print(f"  plain single-lr tensors = {sum(1 for p in model.parameters() if p.requires_grad)} lr={args.lr}")

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
    lr_scheduler = OneCycleLR(
        optimizer,
        total_steps=args.epochs * steps_per_epoch,
        max_lr=args.lr,
        pct_start=0.15,
        div_factor=10,
    )
    print(f"[INFO] OneCycleLR: total_steps={args.epochs * steps_per_epoch}, max_lr={args.lr}, pct_start=0.15, div_factor=10")

    first_batch = next(iter(train_loader))
    print("[INFO] first batch shapes:")
    print("  spikes_data       ", tuple(first_batch["spikes_data"].shape))
    print("  time_attn_mask    ", tuple(first_batch["time_attn_mask"].shape))
    print("  space_attn_mask   ", tuple(first_batch["space_attn_mask"].shape))
    print("  spikes_timestamps ", tuple(first_batch["spikes_timestamps"].shape))
    print("  spikes_spacestamps", tuple(first_batch["spikes_spacestamps"].shape))
    print("  neuron_regions    ", first_batch["neuron_regions"].shape)
    print("  eid[0]            ", first_batch["eid"][0])

    batch = move_batch_to_device(first_batch, device)
    model.eval()
    with torch.no_grad():
        sanity_outputs = model(
            batch["spikes_data"],
            time_attn_mask=batch["time_attn_mask"],
            space_attn_mask=batch["space_attn_mask"],
            spikes_timestamps=batch["spikes_timestamps"],
            spikes_spacestamps=batch["spikes_spacestamps"],
            targets=batch["target"],
            neuron_regions=batch["neuron_regions"],
            masking_mode=args.masking_mode,
            spike_augmentation=False,
            num_neuron=batch["spikes_data"].shape[2],
            eid=batch["eid"][0],
        )
    print(f"[INFO] sanity loss: {sanity_outputs.loss.item():.6f}")
    print(f"[INFO] sanity n_examples: {int(sanity_outputs.n_examples.item())}")

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
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            masking_mode=args.masking_mode,
            train=True,
            max_batches=max_batches,
            lr_scheduler=lr_scheduler,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            masking_mode=args.masking_mode,
            train=False,
            max_batches=max_batches,
        )

        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - training_start_time
        mean_epoch_seconds = elapsed_seconds / max(epoch + 1, 1)
        remaining_epochs = max(args.epochs - epoch - 1, 0)
        eta_seconds = remaining_epochs * mean_epoch_seconds
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss_per_masked_bin={train_loss:.8f} "
            f"val_loss_per_masked_bin={val_loss:.8f} "
            f"epoch_seconds={epoch_seconds:.1f} "
            f"eta_hours={eta_seconds / 3600:.2f}"
        )

        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_checkpoint(args.save_dir, "best", model, optimizer, epoch, best_val_loss)
            print(f"[INFO] new best val loss: {best_val_loss:.8f}")
        else:
            epochs_since_improvement += 1

        save_checkpoint(args.save_dir, "last", model, optimizer, epoch, best_val_loss)
        history.append({
            "epoch": epoch,
            "train_loss_per_masked_bin": train_loss,
            "val_loss_per_masked_bin": val_loss,
            "best_val_loss_so_far": best_val_loss,
        })
        save_history_artifacts(args.save_dir, history)

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print(
                f"[INFO] early stopping at epoch {epoch:03d}: "
                f"no val improvement >= {args.min_delta:g} for {args.patience} epochs"
            )
            break

    print(f"[DONE] Best val loss: {best_val_loss:.8f}")
    print(f"[DONE] Checkpoints saved to: {args.save_dir}")

    if not args.skip_artifacts:
        best_ckpt_path = os.path.join(args.save_dir, "best.pt")
        print(f"[INFO] loading best checkpoint for artifacts: {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        baselines = compute_train_baselines(train_ds)
        split_to_dataset = {"train": train_ds, "val": val_ds, "test": test_ds}
        eval_splits = [args.eval_split]
        for split_name in args.extra_eval_splits:
            if split_name not in eval_splits:
                eval_splits.append(split_name)

        split_summaries = {}
        for split_name in eval_splits:
            eval_ds = split_to_dataset[split_name]
            eval_loader = DataLoader(
                eval_ds,
                batch_size=args.eval_batch_size or args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_local_batch,
            )
            split_summaries[split_name] = evaluate_and_save_artifacts(
                model=model,
                eval_loader=eval_loader,
                device=device,
                save_dir=args.save_dir,
                masking_mode=args.masking_mode,
                eval_split=split_name,
                mask_ratio=args.eval_mask_ratio if args.eval_mask_ratio is not None else args.mask_ratio,
                baselines=baselines,
                use_lograte=bool(config.method.model_kwargs.use_lograte),
                eval_seed=args.artifact_seed,
                max_batches=args.artifact_max_batches,
                save_latents=not args.skip_latent_artifacts,
                artifact_subdir=None if split_name == args.eval_split else f"eval_{split_name}",
            )
        save_split_eval_summary(args.save_dir, split_summaries)


if __name__ == "__main__":
    main()
