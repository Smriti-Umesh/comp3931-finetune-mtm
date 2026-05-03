"""
train_single_session_local_causal.py
Fine-tuning on local session data with causal masking only.
"""

import os
import random
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalSessionDataset(Dataset):
    """
    Local dense single-session dataset.

    Expects files in data_dir:
      spikes_data.npy           [num_windows, T, N]
      time_attn_mask.npy        [num_windows, T]
      space_attn_mask.npy       [num_windows, N]
      spikes_timestamps.npy     [num_windows, T]
      spikes_spacestamps.npy    [num_windows, N]
      neuron_regions.npy        [N]
      eid.npy                   [num_windows]
      split_indices.npz         keys: train_idx, val_idx, test_idx
    """

    def __init__(self, data_dir: str, split: str):
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"Data dir not found: {self.data_dir}"

        split_file = np.load(self.data_dir / "split_indices.npz", allow_pickle=True)
        split_key = {
            "train": "train_idx",
            "val": "val_idx",
            "test": "test_idx",
        }[split]
        self.indices = split_file[split_key].astype(np.int64)

        self.spikes_data = np.load(self.data_dir / "spikes_data.npy", mmap_mode="r")
        self.time_attn_mask = np.load(self.data_dir / "time_attn_mask.npy", mmap_mode="r")
        self.space_attn_mask = np.load(self.data_dir / "space_attn_mask.npy", mmap_mode="r")
        self.spikes_timestamps = np.load(self.data_dir / "spikes_timestamps.npy", mmap_mode="r")
        self.spikes_spacestamps = np.load(self.data_dir / "spikes_spacestamps.npy", mmap_mode="r")
        self.neuron_regions = np.load(self.data_dir / "neuron_regions.npy", allow_pickle=True)
        self.eids = np.load(self.data_dir / "eid.npy", allow_pickle=True)

        # Basic checks
        n_examples, T, N = self.spikes_data.shape
        assert self.time_attn_mask.shape == (n_examples, T)
        assert self.space_attn_mask.shape == (n_examples, N)
        assert self.spikes_timestamps.shape == (n_examples, T)
        assert self.spikes_spacestamps.shape == (n_examples, N)
        assert self.neuron_regions.shape == (N,)
        assert self.eids.shape == (n_examples,)

        self.T = T
        self.N = N

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])

        sample = {
            "spikes_data": torch.tensor(self.spikes_data[i], dtype=torch.float32),            
            "time_attn_mask": torch.tensor(self.time_attn_mask[i], dtype=torch.long),          
            "space_attn_mask": torch.tensor(self.space_attn_mask[i], dtype=torch.long),         
            "spikes_timestamps": torch.tensor(self.spikes_timestamps[i], dtype=torch.long),   
            "spikes_spacestamps": torch.tensor(self.spikes_spacestamps[i], dtype=torch.long),  
            "target": torch.tensor([0.0], dtype=torch.float32),  
            "neuron_regions": np.array(self.neuron_regions, dtype=object),  
            "eid": str(self.eids[i]),
        }
        return sample


def collate_local_batch(batch):
    """
    Keeps eid as list[str].
    Keeps neuron_regions as np.ndarray of shape [B, N].
    """
    out = {
        "spikes_data": torch.stack([x["spikes_data"] for x in batch], dim=0),
        "time_attn_mask": torch.stack([x["time_attn_mask"] for x in batch], dim=0),
        "space_attn_mask": torch.stack([x["space_attn_mask"] for x in batch], dim=0),
        "spikes_timestamps": torch.stack([x["spikes_timestamps"] for x in batch], dim=0),
        "spikes_spacestamps": torch.stack([x["spikes_spacestamps"] for x in batch], dim=0),
        "target": torch.stack([x["target"] for x in batch], dim=0),
        "neuron_regions": np.stack([x["neuron_regions"] for x in batch], axis=0),  # [B, N]
        "eid": [x["eid"] for x in batch],
    }
    return out


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj and hasattr(checkpoint_obj["model"], "state_dict"):
            return checkpoint_obj["model"].state_dict()
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
    if hasattr(checkpoint_obj, "state_dict"):
        return checkpoint_obj.state_dict()
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Unsupported checkpoint format")


def load_pretrained_weights(model, checkpoint_path: str, device: torch.device) -> dict:
    # Keeping full checkpoint loading explicit 
    # because some pretrained artifacts store model objects, not weights only.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = extract_state_dict(checkpoint)
    model_state = model.state_dict()

    loadable_state = {}
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            loadable_state[key] = value
        else:
            skipped_keys.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(loadable_state, strict=False)

    print(f"[INFO] loaded pretrained checkpoint: {checkpoint_path}")
    print(f"[INFO] loaded tensors: {len(loadable_state)}")
    print(f"[INFO] skipped checkpoint tensors: {len(skipped_keys)}")
    print(f"[INFO] missing model tensors after load: {len(missing_keys)}")
    print(f"[INFO] unexpected tensors after load: {len(unexpected_keys)}")

    if skipped_keys:
        print("[INFO] sample skipped keys:", skipped_keys[:10])
    if missing_keys:
        print("[INFO] sample missing keys:", list(missing_keys)[:10])

    # Returning a transfer report so the run artifacts record exactly what was reused.
    return {
        "checkpoint_path": checkpoint_path,
        "loaded_tensor_count": len(loadable_state),
        "skipped_checkpoint_tensor_count": len(skipped_keys),
        "missing_model_tensor_count": len(missing_keys),
        "unexpected_tensor_count": len(unexpected_keys),
        "skipped_keys": skipped_keys,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def cfg_set(config, dotted_path: str, value) -> None:
    # DictConfig.__getattr__ wraps nested dicts in copies
    cur = config
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def build_config(args):
    """
    Build a single-session fine-tuning config that stays close to the
    pretrained IBL MtM setup: stitched backbone + prompt token.
    """
    kwargs = {
        "model": "include:src/configs/ndt1_stitching_prompting.yaml"
    }
    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/ndt1_stitching_prompting.yaml", config)
    config = update_config("src/configs/ssl_session_trainer.yaml", config)

    # Training / logging
    cfg_set(config, "seed", args.seed)
    cfg_set(config, "wandb.use", False)
    cfg_set(config, "wandb.project", "single-session-local")
    cfg_set(config, "training.num_epochs", args.epochs)
    cfg_set(config, "training.train_batch_size", args.batch_size)
    cfg_set(config, "training.test_batch_size", args.batch_size)

    # Data
    cfg_set(config, "data.dataset_name", "local_ssl")
    cfg_set(config, "data.max_time_length", args.time_bins)
    cfg_set(config, "data.max_space_length", args.num_neurons)
    cfg_set(config, "data.load_meta", True)
    cfg_set(config, "data.spike_augmentation", False)
    cfg_set(config, "data.target", None)

    # Model
    cfg_set(config, "model.model_class", "NDT1")
    cfg_set(config, "model.encoder.stitching", True)

    # Masking: causal only
    cfg_set(config, "model.encoder.masker.force_active", True)
    cfg_set(config, "model.encoder.masker.mode", "causal")
    cfg_set(config, "model.encoder.masker.ratio", args.mask_ratio)
    cfg_set(config, "model.encoder.masker.zero_ratio", 1.0)
    cfg_set(config, "model.encoder.masker.random_ratio", 0.0)
    cfg_set(config, "model.encoder.masker.expand_prob", 0.0)
    cfg_set(config, "model.encoder.masker.max_timespan", 1)
    cfg_set(config, "model.encoder.masker.mask_regions", ["all"])
    cfg_set(config, "model.encoder.masker.target_regions", ["all"])
    cfg_set(config, "model.encoder.masker.n_mask_regions", 1)

    # Causal attention context: no future tokens, unlimited past.
    cfg_set(config, "model.encoder.context.forward", 0)
    cfg_set(config, "model.encoder.context.backward", -1)

    # Keeping the pretrained shared stitched width; 
    # as only time length is data-specific.
    cfg_set(config, "model.encoder.embedder.max_F", args.time_bins)
    cfg_set(config, "model.encoder.embedder.use_prompt", True)
    cfg_set(config, "model.encoder.embedder.use_session", True)

    # Optimizer
    cfg_set(config, "optimizer.lr", args.lr)
    cfg_set(config, "optimizer.wd", args.weight_decay)
    cfg_set(config, "optimizer.eps", 1e-8)

    return config


def make_model(config, num_neurons: int):
    meta_data = {
        # Stitching still needs the local raw neuron count so it can build the
        # single-session input/output projection modules.
        "num_neurons": [num_neurons],
    }
    model = NDT1(
        config.model,
        **config.method.model_kwargs,
        **meta_data,
    )
    return model


def run_epoch(model, loader, optimizer, device, train: bool, max_batches: int = None, lr_scheduler=None):
    if train:
        model.train()
    else:
        model.eval()

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
                masking_mode="causal",
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


def save_checkpoint(save_dir, name, model, optimizer, epoch, best_val_loss):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, os.path.join(save_dir, f"{name}.pt"))


def make_optimizer(model, num_neurons: int, base_lr: float, adapter_lr: float | None, weight_decay: float, eps: float):
    if adapter_lr is None:
        # Plain fine-tuning baseline: every trainable parameter uses the same LR,
        params = [param for param in model.parameters() if param.requires_grad]
        param_count = sum(param.numel() for param in params)
        print("[INFO] optimizer parameter groups:")
        print(f"  plain single-lr tensors = {len(params)} params={param_count} lr={base_lr}")
        return AdamW(params, lr=base_lr, weight_decay=weight_decay, eps=eps)

    # The 355-neuron stitcher/decoder are newly initialized for this local session,
    adapter_prefixes = (
        f"encoder.stitcher.stitcher_dict.{num_neurons}.",
        f"stitch_decoder.stitch_decoder_dict.{num_neurons}.",
    )
    backbone_params = []
    adapter_params = []
    adapter_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in adapter_prefixes):
            adapter_params.append(param)
            adapter_names.append(name)
        else:
            backbone_params.append(param)

    param_groups = [{"params": backbone_params, "lr": base_lr}]
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": adapter_lr})
    else:
        print("[WARN] no local stitcher/decoder adapter params matched; using backbone LR for all params")

    backbone_count = sum(p.numel() for p in backbone_params)
    adapter_count = sum(p.numel() for p in adapter_params)
    print("[INFO] optimizer parameter groups:")
    print(f"  backbone/shared tensors = {len(backbone_params)} params={backbone_count} lr={base_lr}")
    print(f"  local adapter tensors   = {len(adapter_params)} params={adapter_count} lr={adapter_lr}")
    if adapter_names:
        print(f"  local adapter sample    = {adapter_names[:4]}")

    return AdamW(param_groups, weight_decay=weight_decay, eps=eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Absolute path to the preprocessed control-session directory")
    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="Path to pretrained IBL MtM checkpoint (.pt)")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="LR for transferred/shared pretrained parameters")
    parser.add_argument("--adapter-lr", type=float, default=None,
                        help="Optional LR for local 355-neuron stitcher/decoder parameters; omit for plain single-LR fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
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
                        help="Held-out future-time ratio for artifact evaluation; defaults to --mask-ratio")
    parser.add_argument("--artifact-max-batches", type=int, default=None,
                        help="Optional cap for quick artifact generation during smoke jobs")
    parser.add_argument("--artifact-seed", type=int, default=123,
                        help="Seed for deterministic held-out artifact masks")
    parser.add_argument("--skip-artifacts", action="store_true",
                        help="Skip post-training prediction tables and plots")
    parser.add_argument("--skip-latent-artifacts", action="store_true",
                        help="Skip latent PCA artifacts while keeping prediction artifacts")
    args = parser.parse_args()
    print("[CONFIG]")
    print(f"  save_dir     = {args.save_dir}")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  epochs       = {args.epochs}")
    print(f"  lr           = {args.lr}")
    print(f"  adapter_lr   = {args.adapter_lr}")
    print(f"  weight_decay = {args.weight_decay}")
    print(f"  mask_ratio   = {args.mask_ratio}")
    print(f"  patience     = {args.patience}")
    print(f"  min_delta    = {args.min_delta}")
    print(f"  smoke_test   = {args.smoke_test}")

    set_seed(args.seed)

    # Loading one split first to infer dimensions
    train_ds = LocalSessionDataset(args.data_dir, split="train")
    val_ds = LocalSessionDataset(args.data_dir, split="val")
    # Loading the test split 
    test_ds = LocalSessionDataset(args.data_dir, split="test")

    args.time_bins = train_ds.T
    args.num_neurons = train_ds.N
    if args.save_dir is None:
        run_type = "smoke" if args.smoke_test else "full"
        args.save_dir = f"results_20ms/causal_{run_type}_bs{args.batch_size}_ep{args.epochs}"

    print(f"[INFO] train windows: {len(train_ds)}")
    print(f"[INFO] val windows:   {len(val_ds)}")
    print(f"[INFO] test windows:  {len(test_ds)}")
    print(f"[INFO] shape per example: [{train_ds.T}, {train_ds.N}]")

    # Building config and model
    config = build_config(args)
    print("[CONFIG]")
    print(f"  pretrained_ckpt = {args.pretrained_ckpt}")
    print(f"  masking_mode    = causal")
    print(f"  mask_ratio      = {config.model.encoder.masker.ratio}")
    print(f"  stitching       = {config.model.encoder.stitching}")
    print(f"  use_prompt      = {config.model.encoder.embedder.use_prompt}")
    print(f"  use_session     = {config.model.encoder.embedder.use_session}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  epochs          = {args.epochs}")
    print(f"  lr              = {args.lr}")
    print(f"  adapter_lr      = {args.adapter_lr}")
    print(f"  patience        = {args.patience}")
    print(f"  min_delta       = {args.min_delta}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    model = make_model(config, num_neurons=train_ds.N).to(device)
    load_report = load_pretrained_weights(model, args.pretrained_ckpt, device)
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
        pct_start=0.2,
        div_factor=10,
    )
    print(f"[INFO] OneCycleLR: total_steps={args.epochs * steps_per_epoch}, max_lr={max_lr}, pct_start=0.2, div_factor=10")

    # Smoke-test one batch before any full training
    first_batch = next(iter(train_loader))
    print("[INFO] first batch shapes:")
    print("  spikes_data       ", tuple(first_batch["spikes_data"].shape))
    print("  time_attn_mask    ", tuple(first_batch["time_attn_mask"].shape))
    print("  space_attn_mask   ", tuple(first_batch["space_attn_mask"].shape))
    print("  spikes_timestamps ", tuple(first_batch["spikes_timestamps"].shape))
    print("  spikes_spacestamps", tuple(first_batch["spikes_spacestamps"].shape))
    print("  neuron_regions    ", first_batch["neuron_regions"].shape)
    print("  eid[0]            ", first_batch["eid"][0])

    # One forward pass sanity check
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
            masking_mode="causal",
            spike_augmentation=False,
            num_neuron=batch["spikes_data"].shape[2],
            eid=batch["eid"][0],
        )
    print(f"[INFO] sanity loss: {sanity_outputs.loss.item():.6f}")
    print(f"[INFO] sanity n_examples: {int(sanity_outputs.n_examples.item())}")

    max_batches = 3 if args.smoke_test else None

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    # Save run metadata before the long loop so failed jobs still have provenance.
    save_run_metadata(
        args.save_dir,
        args,
        config,
        {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    )
    save_pretrained_load_report(args.save_dir, load_report)
    history = []
    training_start_time = time.perf_counter()
    # Early stopping is intentionally based on validation loss, not train loss,
    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            max_batches=max_batches,
            lr_scheduler=lr_scheduler,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
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

        # Saving `last.pt` after best-val is recorded
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
        # Reloading the best checkpoint so final plots/tables correspond to best model.
        best_ckpt_path = os.path.join(args.save_dir, "best.pt")
        print(f"[INFO] loading best checkpoint for artifacts: {best_ckpt_path}")
        # Keeping full checkpoint loading
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
                masking_mode="causal",
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
