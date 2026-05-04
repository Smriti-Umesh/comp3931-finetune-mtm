from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

# Path - change accordingly to your setup

# Raw Kilosort folder - symlink one
KS_DIR = Path("data/raw/control_session_kilosort4").expanduser()

# Path for processed output to be saved
OUT_DIR = Path("data/control_session_preprocessed_new").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Session ID string saved with every window
EID = "control_session"

# Recording/sample settings
SAMPLE_RATE = 30000.0     # Hz
BIN_SIZE_MS = 20.0        # 20 ms bins
WINDOW_SIZE_S = 2.0       # each training example is 2 seconds long
STRIDE_S = 1.0            # moving window by 1 second each time

# Split settings. split the continuous
# session first, then window each split separately so train/val/test do not
# share overlapping raw bins across split units.
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
TEST_FRAC = 0.2
SPLIT_GAP_S = 20.0

# Keep overlap in training to preserve sample count, but make validation/test
# windows non-overlapping so eval is stricter.
TRAIN_STRIDE_S = STRIDE_S
EVAL_STRIDE_S = WINDOW_SIZE_S

# Unit filtering
UNIT_LABEL_TO_KEEP = "good"

# Storage dtype for spike counts
SPIKE_DTYPE = np.uint16


def find_cluster_table(ks_dir: Path) -> Path:

    for name in ["cluster_group.tsv", "cluster_info.tsv"]:
        path = ks_dir / name
        if path.exists():
            return path
    raise FileNotFoundError("Could not find cluster_group.tsv or cluster_info.tsv")


def load_cluster_labels(cluster_table_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with standardized columns:
      - cluster_id
      - unit_label
    """
    df = pd.read_csv(cluster_table_path, sep="\t")

    cols_lower = {c.lower(): c for c in df.columns}

    cluster_col = None
    for cand in ["cluster_id", "id", "clusterid"]:
        if cand in cols_lower:
            cluster_col = cols_lower[cand]
            break
    if cluster_col is None:
        raise ValueError(f"Could not find cluster id column in {cluster_table_path.name}")

    label_col = None
    for cand in ["group", "kslabel", "label"]:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break
    if label_col is None:
        raise ValueError(f"Could not find label column in {cluster_table_path.name}")

    out = df.rename(columns={cluster_col: "cluster_id", label_col: "unit_label"}).copy()
    out["cluster_id"] = out["cluster_id"].astype(int)
    out["unit_label"] = out["unit_label"].astype(str)
    return out[["cluster_id", "unit_label"]]


def build_binned_spike_matrix(
    spike_times_samples: np.ndarray,
    spike_clusters: np.ndarray,
    kept_units: np.ndarray,
    sample_rate: float,
    bin_size_ms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build one full continuous spike-count matrix of shape (total_bins, N)

    Rows    = time bins
    Columns = neurons
    """
    if len(spike_times_samples) != len(spike_clusters):
        raise ValueError("spike_times and spike_clusters must have same length")

    # Convert bin size from ms to seconds
    bin_size_s = bin_size_ms / 1000.0

    # Recording duration in seconds
    duration_s = float(spike_times_samples.max() / sample_rate)

    # Number of bins across the whole session
    total_bins = int(math.ceil(duration_s / bin_size_s))

    # Sort units so column order is stable
    unit_ids = np.asarray(sorted(map(int, kept_units)), dtype=np.int32)
    n_units = len(unit_ids)

    # Map each unit ID to a column index in the matrix
    unit_to_col = {int(unit_id): col_idx for col_idx, unit_id in enumerate(unit_ids)}

    # Keep only spikes belonging to kept units
    keep_mask = np.isin(spike_clusters, unit_ids)
    st = spike_times_samples[keep_mask].astype(np.int64)
    sc = spike_clusters[keep_mask].astype(np.int32)

    # Convert spike sample index -> spike time in seconds -> bin index
    bin_idx = np.floor((st / sample_rate) / bin_size_s).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, total_bins - 1)

    # Convert cluster ID -> matrix column index
    col_idx = np.fromiter((unit_to_col[int(c)] for c in sc), dtype=np.int64, count=len(sc))

    # Count spikes into a 2D matrix efficiently
    flat_idx = bin_idx * n_units + col_idx
    counts = np.bincount(flat_idx, minlength=total_bins * n_units)
    full_matrix = counts.reshape(total_bins, n_units)

    return full_matrix, unit_ids, duration_s


def sliding_windows(matrix: np.ndarray, window_bins: int, stride_bins: int) -> np.ndarray:
    """
    Cut one long (total_bins, N) matrix into overlapping windows
    of shape (num_windows, window_bins, N)
    """
    total_bins, n_units = matrix.shape

    if total_bins < window_bins:
        raise ValueError(
            f"Not enough bins ({total_bins}) for one window of size {window_bins}"
        )

    starts = np.arange(0, total_bins - window_bins + 1, stride_bins, dtype=np.int64)
    windows = np.empty((len(starts), window_bins, n_units), dtype=matrix.dtype)

    for i, s in enumerate(starts):
        windows[i] = matrix[s:s + window_bins]

    return windows


def build_temporally_stronger_split_windows(
    matrix: np.ndarray,
    window_bins: int,
    train_stride_bins: int,
    eval_stride_bins: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_gap_bins: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """
    Split the full continuous matrix first, then create windows inside each
    split separately.

    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Train/val/test fractions must sum to 1.0")

    total_bins = matrix.shape[0]
    train_end_bin = int(total_bins * train_frac)
    val_end_bin = int(total_bins * (train_frac + val_frac))

    # Leave a real temporal gap before validation and test so nearby bins do
    # not leak across split boundaries through overlapping windows.
    val_start_bin = train_end_bin + split_gap_bins
    test_start_bin = val_end_bin + split_gap_bins

    train_matrix = matrix[:train_end_bin]
    val_matrix = matrix[val_start_bin:val_end_bin]
    test_matrix = matrix[test_start_bin:]

    split_matrices = {
        "train": train_matrix,
        "val": val_matrix,
        "test": test_matrix,
    }
    for split_name, split_matrix in split_matrices.items():
        if split_matrix.shape[0] < window_bins:
            raise ValueError(
                f"Split '{split_name}' has only {split_matrix.shape[0]} bins after "
                f"applying temporal gaps, which is not enough for one window of "
                f"size {window_bins}."
            )

    # Training keeps overlap for data efficiency. Validation and test use
    # stride == window size so each eval window is distinct in raw time.
    train_windows = sliding_windows(train_matrix, window_bins, train_stride_bins)
    val_windows = sliding_windows(val_matrix, window_bins, eval_stride_bins)
    test_windows = sliding_windows(test_matrix, window_bins, eval_stride_bins)

    windows = np.concatenate([train_windows, val_windows, test_windows], axis=0)

    n_train = len(train_windows)
    n_val = len(val_windows)
    n_test = len(test_windows)
    split_idx = {
        "train_idx": np.arange(0, n_train, dtype=np.int64),
        "val_idx": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test_idx": np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64),
    }

    split_metadata = {
        "total_bins": int(total_bins),
        "train_end_bin": int(train_end_bin),
        "val_start_bin": int(val_start_bin),
        "val_end_bin": int(val_end_bin),
        "test_start_bin": int(test_start_bin),
        "split_gap_bins": int(split_gap_bins),
        "train_bins": int(train_matrix.shape[0]),
        "val_bins": int(val_matrix.shape[0]),
        "test_bins": int(test_matrix.shape[0]),
        "train_windows": int(n_train),
        "val_windows": int(n_val),
        "test_windows": int(n_test),
    }
    return windows, split_idx, split_metadata


# loading kilosort outputs

print("Loading raw Kilosort files...")
spike_times = np.load(KS_DIR / "spike_times.npy", mmap_mode="r")
spike_clusters = np.load(KS_DIR / "spike_clusters.npy", mmap_mode="r")

cluster_table_path = find_cluster_table(KS_DIR)
cluster_table = load_cluster_labels(cluster_table_path)

print(f"spike_times shape: {spike_times.shape}")
print(f"spike_clusters shape: {spike_clusters.shape}")
print(f"cluster table file: {cluster_table_path.name}")


# keeping only "good" units as defined by the cluster table labels

kept_units = cluster_table.loc[
    cluster_table["unit_label"].str.lower() == UNIT_LABEL_TO_KEEP.lower(),
    "cluster_id"
].to_numpy(dtype=np.int32)

if kept_units.size == 0:
    raise ValueError(f"No units found with label '{UNIT_LABEL_TO_KEEP}'")

print(f"Total units labeled '{UNIT_LABEL_TO_KEEP}': {len(kept_units)}")


# (total_bins, N) full binned spike matrix for the entire session, keeping only "good" units

print("Building full binned spike matrix...")
full_matrix, unit_ids, duration_s = build_binned_spike_matrix(
    spike_times_samples=np.asarray(spike_times),
    spike_clusters=np.asarray(spike_clusters),
    kept_units=kept_units,
    sample_rate=SAMPLE_RATE,
    bin_size_ms=BIN_SIZE_MS,
)

if full_matrix.max() > np.iinfo(np.uint16).max:
    raise ValueError("Spike counts exceed uint16 range")

full_matrix = full_matrix.astype(SPIKE_DTYPE, copy=False)

print(f"Recording duration (s): {duration_s:.3f}")
print(f"Full matrix shape: {full_matrix.shape}")   # (total_bins, N)


# Convert the continuous matrix into split-specific windows. 
# train/val/test are now separated in continuous time before windowing.

bin_size_s = BIN_SIZE_MS / 1000.0
window_bins = int(round(WINDOW_SIZE_S / bin_size_s))
train_stride_bins = int(round(TRAIN_STRIDE_S / bin_size_s))
eval_stride_bins = int(round(EVAL_STRIDE_S / bin_size_s))
split_gap_bins = int(round(SPLIT_GAP_S / bin_size_s))

print(f"Window bins: {window_bins}")
print(f"Train stride bins: {train_stride_bins}")
print(f"Eval stride bins: {eval_stride_bins}")
print(f"Split gap bins: {split_gap_bins}")

windows, split_idx, split_metadata = build_temporally_stronger_split_windows(
    matrix=full_matrix,
    window_bins=window_bins,
    train_stride_bins=train_stride_bins,
    eval_stride_bins=eval_stride_bins,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
    split_gap_bins=split_gap_bins,
)

n_windows, T, N = windows.shape
print(f"Windowed spikes_data shape: {windows.shape}")   # (num_windows, T, N)
print(
    "Window counts by split: "
    f"train={split_metadata['train_windows']}, "
    f"val={split_metadata['val_windows']}, "
    f"test={split_metadata['test_windows']}"
)


# arrays for attention masks, timestamps, spacestamps, and metadata to be saved with the windows

# Since every window is full-length and every kept neuron is valid,
# these masks are all ones.
time_attn_mask = np.ones((n_windows, T), dtype=np.int8)
space_attn_mask = np.ones((n_windows, N), dtype=np.int8)

# Timestamps within each window: 0, 1, 2, ..., T-1
spikes_timestamps = np.broadcast_to(np.arange(T, dtype=np.int64), (n_windows, T)).copy()

# Neuron indices within each window: 0, 1, 2, ..., N-1
spikes_spacestamps = np.broadcast_to(np.arange(N, dtype=np.int64), (n_windows, N)).copy()

# Same session ID repeated for every window
eid_array = np.asarray([EID] * n_windows, dtype=object)

# Placeholder regions because we do not have region annotations yet
neuron_regions = np.asarray(["unknown"] * N, dtype=object)


print("Saving outputs...")

np.save(OUT_DIR / "spikes_data.npy", windows)
np.save(OUT_DIR / "time_attn_mask.npy", time_attn_mask)
np.save(OUT_DIR / "spikes_timestamps.npy", spikes_timestamps)
np.save(OUT_DIR / "spikes_spacestamps.npy", spikes_spacestamps)
np.save(OUT_DIR / "space_attn_mask.npy", space_attn_mask)
np.save(OUT_DIR / "eid.npy", eid_array)
np.save(OUT_DIR / "unit_ids.npy", unit_ids)
np.save(OUT_DIR / "neuron_regions.npy", neuron_regions)
np.savez(OUT_DIR / "split_indices.npz", **split_idx)


metadata = {
    "eid": EID,
    "ks_dir": str(KS_DIR),
    "out_dir": str(OUT_DIR),
    "sample_rate_hz": SAMPLE_RATE,
    "bin_size_ms": BIN_SIZE_MS,
    "window_size_s": WINDOW_SIZE_S,
    "stride_s": STRIDE_S,
    "train_stride_s": TRAIN_STRIDE_S,
    "eval_stride_s": EVAL_STRIDE_S,
    "split_gap_s": SPLIT_GAP_S,
    "train_frac": TRAIN_FRAC,
    "val_frac": VAL_FRAC,
    "test_frac": TEST_FRAC,
    "duration_s": float(duration_s),
    "num_good_units": int(N),
    "num_windows": int(n_windows),
    "window_shape": [int(T), int(N)],
    "split_metadata": split_metadata,
}

with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done.")
print(json.dumps(metadata, indent=2))
