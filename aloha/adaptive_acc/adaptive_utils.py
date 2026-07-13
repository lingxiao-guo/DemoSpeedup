import numpy as np
import torch


def extract_action_chunk(actions, start_idx, length, pad_last_frame=True):
    total_frames = len(actions)
    actual_end = min(start_idx + length, total_frames)
    chunk = actions[start_idx:actual_end].astype(np.float32)

    if len(chunk) < length and pad_last_frame:
        pad_len = length - len(chunk)
        last_frame = actions[-1:].astype(np.float32)
        chunk = np.concatenate([chunk, np.repeat(last_frame, pad_len, axis=0)], axis=0)

    return chunk


def resample_linear(actions, original_length, target_length):
    if original_length == target_length:
        return actions.copy()
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    resampled = np.zeros((target_length, actions.shape[1]), dtype=np.float32)
    for dim in range(actions.shape[1]):
        resampled[:, dim] = np.interp(target_indices, original_indices, actions[:, dim])
    return resampled


def process_action_adaptive(action_data, factor, chunk_size, action_len=None):
    action_np = action_data.cpu().numpy() if isinstance(action_data, torch.Tensor) else action_data
    L = max(1, int(chunk_size * factor))
    if action_len is not None and L > action_len:
        L = max(1, action_len)
    chunk = extract_action_chunk(action_np, 0, L, pad_last_frame=True)
    if L == chunk_size:
        resampled = chunk
    else:
        resampled = resample_linear(chunk, L, chunk_size)
    action_data[:chunk_size] = torch.from_numpy(resampled).float().to(action_data.device)
    return action_data


def load_factors_hdf5(dataset_path, chunk_size, dtw_threshold):
    try:
        import h5py
        threshold_str = str(dtw_threshold).replace(".", "_")
        field_name = f"adaptive_factors_{chunk_size}_{threshold_str}"
        with h5py.File(dataset_path, "r") as f:
            if field_name in f:
                return np.array(f[field_name][()], dtype=np.float64)
    except Exception:
        pass
    return None


def write_factors_hdf5(dataset_path, factors, chunk_size, dtw_threshold):
    import h5py
    threshold_str = str(dtw_threshold).replace(".", "_")
    field_name = f"adaptive_factors_{chunk_size}_{threshold_str}"

    with h5py.File(dataset_path, "r+") as f:
        f[field_name] = factors.astype(np.float64)
