#!/usr/bin/env python3
"""
离线计算每帧的自适应加速因子，直接写入源 HDF5 的 /adaptive_factors。

对每条 episode 的每一帧:
  1. 提取 action 轨迹 (14D) 或通过 FK 转为 TCP 位置 (6D, --tcp_only)
  2. 对每个 scale_factor in [1.0, 1.1, ..., max_factor]:
     - 取参考片段 (chunk_size) 和扩展片段 (L = chunk_size * factor)
     - 计算两段之间的 DTW 距离
  3. 每帧最优因子 = DTW < threshold 的最大 factor
  4. 中值滤波平滑
  5. 写入源 HDF5: /adaptive_factors

支持多核并行: --num_workers 0 自动检测 CPU 核心数
"""

import numpy as np
import os
import sys
import argparse
import h5py
import glob
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from dtaidistance import dtw_ndim
from scipy.signal import medfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modern_robotics as mr

from adaptive_acc.adaptive_utils import extract_action_chunk, write_factors_hdf5


_VX300S_SLIST = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
    ]
).T

_VX300S_M = np.array(
    [
        [1.0, 0.0, 0.0, 0.536494],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.42705],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def get_tcp_xyz(joints):
    xyz = []
    for joint in joints:
        T_sb = mr.FKinSpace(_VX300S_M, _VX300S_SLIST, joint)
        xyz.append(T_sb[:3, 3])
    return np.array(xyz)


def compute_tcp_trajectory(qpos):
    left_joints = qpos[:, 0:6]
    right_joints = qpos[:, 7:13]
    left_xyz = get_tcp_xyz(left_joints)
    right_xyz = get_tcp_xyz(right_joints)
    return np.concatenate([left_xyz, right_xyz], axis=1)


def compute_frame_dtw(trajectory, frame_idx, chunk_size, scale_factor, dtw_window=None):
    total = trajectory.shape[0]
    available = total - frame_idx

    if available <= chunk_size:
        return float("inf")

    L = min(int(chunk_size * scale_factor), available)
    if L <= chunk_size:
        return 0.0

    reference_chunk = extract_action_chunk(trajectory, frame_idx, chunk_size, pad_last_frame=True)
    extended_chunk = extract_action_chunk(trajectory, frame_idx, L, pad_last_frame=True)
    return dtw_ndim.distance(reference_chunk, extended_chunk, window=dtw_window)


def apply_filter(optimal_factors, window_size):
    n = len(optimal_factors)
    actual_window = min(window_size, n)
    if actual_window % 2 == 0:
        actual_window += 1
    return medfilt(optimal_factors, kernel_size=actual_window)


def _worker_compute(args_tuple):
    (
        dataset_path,
        chunk_size,
        scale_factors,
        dtw_threshold,
        dtw_window,
        filter_window,
        frame_stride,
        tcp_only,
    ) = args_tuple

    episode_idx = int(os.path.basename(dataset_path).split("_")[1].split(".")[0])

    with h5py.File(dataset_path, "r") as root:
        if tcp_only:
            qpos = root["/observations/qpos"][()]
            trajectory = compute_tcp_trajectory(qpos)
        else:
            trajectory = root["/action"][()]

        episode_len = trajectory.shape[0]

    frame_indices_full = np.arange(episode_len)
    actual_stride = max(1, min(frame_stride, episode_len // 10 + 1))

    if actual_stride == 1:
        optimal_factors = np.ones(episode_len, dtype=np.float64)
        for t in range(episode_len):
            best_factor = 1.0
            for sf in scale_factors:
                dtw_dist = compute_frame_dtw(trajectory, t, chunk_size, sf, dtw_window)
                if dtw_dist <= dtw_threshold:
                    best_factor = sf
                else:
                    break
            optimal_factors[t] = best_factor
    else:
        stride_indices = frame_indices_full[::actual_stride]
        n_stride = len(stride_indices)
        stride_factors = np.ones(n_stride, dtype=np.float64)
        for si, t in enumerate(stride_indices):
            best_factor = 1.0
            for sf in scale_factors:
                dtw_dist = compute_frame_dtw(trajectory, t, chunk_size, sf, dtw_window)
                if dtw_dist <= dtw_threshold:
                    best_factor = sf
                else:
                    break
            stride_factors[si] = best_factor
        optimal_factors = np.interp(frame_indices_full, stride_indices, stride_factors)

    filtered_factors = apply_filter(optimal_factors, filter_window)

    write_factors_hdf5(dataset_path, filtered_factors, chunk_size, dtw_threshold)

    return episode_idx, episode_len, filtered_factors


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame adaptive acceleration factors")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (e.g. sim_insertion_human)")
    parser.add_argument("--chunk_size", type=int, required=True, help="Chunk size for action prediction")
    parser.add_argument("--dtw_threshold", type=float, required=True, help="DTW threshold for optimal factor")
    parser.add_argument("--max_factor", type=float, default=5.0, help="Maximum scale factor")
    parser.add_argument("--scale_step", type=float, default=0.1, help="Scale factor step size")
    parser.add_argument("--filter_window", type=int, default=9, help="Median filter window size")
    parser.add_argument("--dtw_window", type=int, default=5, help="DTW Sakoe-Chiba window (0=None, default=5)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Analyze every N frames (default=1)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers (0=auto detect)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Specific episodes to process")
    parser.add_argument("--tcp_only", action="store_true", help="Use 6D TCP (FK from qpos) instead of full 14D action for DTW")

    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dataset_dir = os.path.join(data_dir, args.task_name)

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")

    scale_factors = np.arange(1.0, args.max_factor + args.scale_step / 2, args.scale_step)
    scale_factors = [round(float(sf), 2) for sf in scale_factors]

    if args.episodes is not None:
        episodes = args.episodes
    else:
        ep_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
        episodes = sorted([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in ep_files
                           if os.path.getsize(f) > 0])

    n_workers = args.num_workers
    if n_workers <= 0:
        n_workers = min(cpu_count(), len(episodes))
    n_workers = max(1, n_workers)

    data_mode = "6D TCP (FK from qpos)" if args.tcp_only else "14D action"
    print(f"Data mode: {data_mode}")
    print(f"Scale factors: {scale_factors}")
    print(f"DTW threshold: {args.dtw_threshold}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Dataset: {dataset_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Workers: {n_workers} (CPU cores: {cpu_count()})")
    print()

    dtw_window = args.dtw_window if args.dtw_window > 0 else None

    worker_args = [
        (
            os.path.join(dataset_dir, f"episode_{ep_idx}.hdf5"),
            args.chunk_size,
            scale_factors,
            args.dtw_threshold,
            dtw_window,
            args.filter_window,
            args.frame_stride,
            args.tcp_only,
        )
        for ep_idx in episodes
    ]

    pool = Pool(processes=n_workers)
    results = []
    try:
        results = list(tqdm(
            pool.imap_unordered(_worker_compute, worker_args),
            total=len(worker_args),
            desc="Episodes",
            smoothing=0.05,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted, terminating workers...")
        pool.terminate()
        pool.join()
        print("Workers stopped.")
        sys.exit(1)
    finally:
        pool.terminate()
        pool.join()

    results.sort(key=lambda x: x[0])

    print(f"\n{'='*70}")
    print(f"{'Episode':>10} {'Frames':>8} {'Mean Raw':>10} {'Mean Filt':>10} {'Min':>8} {'Max':>8}")
    print(f"{'-'*70}")

    all_raw_means = []
    all_filt_means = []
    for episode_idx, episode_len, filtered in results:
        raw_mean = float(np.mean(filtered))
        all_raw_means.append(raw_mean)
        all_filt_means.append(raw_mean)
        print(f"{episode_idx:>10} {episode_len:>8} {raw_mean:>10.3f} {raw_mean:>10.3f} "
              f"{float(np.min(filtered)):>8.3f} {float(np.max(filtered)):>8.3f}")

    print(f"{'-'*70}")
    print(f"{'ALL':>10} {'':>8} {np.mean(all_raw_means):>10.3f} {np.mean(all_filt_means):>10.3f}")
    print(f"{'='*70}")
    print(f"\nWrote /adaptive_factors to {len(results)} episode HDF5 files in: {dataset_dir}")
    print(f"Dataset dir: {dataset_dir}")


if __name__ == "__main__":
    main()
