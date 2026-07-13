#!/usr/bin/env python3
"""
将 HDF5 episode 渲染为视频，叠加：
  - 右上角：因子曲线 + 当前帧指示
  - 左下角：3D TCP 轨迹 (左臂 + 右臂) + 当前帧指示

用法:
  python adaptive_acc/visualize.py \
      --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
      --output_dir ./videos \
      --camera top --chunk_size 50 --fps 50
"""

import numpy as np
import cv2
import os
import sys
import argparse
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modern_robotics as mr
from adaptive_acc.adaptive_utils import load_factors_hdf5


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


def compute_chained_factors(optimal_factors, start_frame, chunk_size):
    total_frames = len(optimal_factors)
    chain = []
    current = start_frame

    while current < total_frames:
        factor = optimal_factors[min(current, total_frames - 1)]
        chain.append((current, factor))
        step = int(chunk_size * factor)
        step = max(1, step)
        current = current + step

    return chain


class FactorCurveOverlay:
    def __init__(self, factors, chunk_size, width=240, height=140, dpi=100):
        self.factors = factors
        self.chunk_size = chunk_size
        self.n_frames = len(factors)
        self.width = width
        self.height = height

        self.fig, self.ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        frame_indices = np.arange(self.n_frames)
        self.line, = self.ax.plot(frame_indices, factors, color="#1f77b4", linewidth=1.5)
        self.vline = self.ax.axvline(x=0, color="red", linewidth=2, linestyle="--")

        chain = compute_chained_factors(factors, 0, chunk_size)
        self.chain_frames = [c[0] for c in chain]
        self.chain_factors_vals = [c[1] for c in chain]
        self.ax.scatter(self.chain_frames, self.chain_factors_vals,
                        color="#d62728", s=12, marker="D", zorder=5)

        self.ax.set_xlim(0, self.n_frames - 1)
        ymin = max(0.9, np.min(factors) * 0.95)
        ymax = np.max(factors) * 1.05
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor((0.97, 0.97, 0.97, 0.85))
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)
        self.fig.patch.set_facecolor((1, 1, 1, 0.85))
        self.fig.canvas.draw()

    def render(self, t):
        self.vline.set_xdata([t, t])
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        h, w = img.shape[:2]
        if (h, w) != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return img


class TCP3DOverlay:
    def __init__(self, tcp, factors, chunk_size, width=320, height=160, dpi=100):
        self.tcp = tcp
        self.left_ee = tcp[:, 0:3]
        self.right_ee = tcp[:, 3:6]
        self.n_frames = len(tcp)
        self.width = width
        self.height = height

        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        self.ax1 = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax1.plot(self.left_ee[:, 0], self.left_ee[:, 1], self.left_ee[:, 2],
                      color="steelblue", linewidth=0.8, alpha=0.7)
        self.ax1.scatter(self.left_ee[0, 0], self.left_ee[0, 1], self.left_ee[0, 2],
                         color="green", s=20, marker="o")
        self.ax1.scatter(self.left_ee[-1, 0], self.left_ee[-1, 1], self.left_ee[-1, 2],
                         color="gray", s=15, marker="X")
        self.left_pt = self.ax1.scatter([], [], [], color="red", s=40, marker="o", zorder=10)
        self.ax1.set_title("Left Arm", fontsize=7)
        self._set_3d_limits(self.ax1, self.left_ee)
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_zticklabels([])

        self.ax2 = self.fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2.plot(self.right_ee[:, 0], self.right_ee[:, 1], self.right_ee[:, 2],
                      color="darkorange", linewidth=0.8, alpha=0.7)
        self.ax2.scatter(self.right_ee[0, 0], self.right_ee[0, 1], self.right_ee[0, 2],
                         color="green", s=20, marker="o")
        self.ax2.scatter(self.right_ee[-1, 0], self.right_ee[-1, 1], self.right_ee[-1, 2],
                         color="gray", s=15, marker="X")
        self.right_pt = self.ax2.scatter([], [], [], color="red", s=40, marker="o", zorder=10)
        self.ax2.set_title("Right Arm", fontsize=7)
        self._set_3d_limits(self.ax2, self.right_ee)
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_zticklabels([])

        if factors is not None:
            chain = compute_chained_factors(factors, 0, chunk_size)
            self.chain_indices = [c[0] for c in chain]
            chain_left = [self.left_ee[i] for i in self.chain_indices if i < self.n_frames]
            chain_right = [self.right_ee[i] for i in self.chain_indices if i < self.n_frames]
            if chain_left:
                cl = np.array(chain_left)
                self.ax1.scatter(cl[:, 0], cl[:, 1], cl[:, 2],
                                 color="#d62728", s=10, marker="D", zorder=8)
            if chain_right:
                cr = np.array(chain_right)
                self.ax2.scatter(cr[:, 0], cr[:, 1], cr[:, 2],
                                 color="#d62728", s=10, marker="D", zorder=8)

        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.15)
        self.fig.patch.set_facecolor((1, 1, 1, 0.85))
        self.fig.canvas.draw()

    def _set_3d_limits(self, ax, ee):
        margin = 0.02
        ax.set_xlim(np.min(ee[:, 0]) - margin, np.max(ee[:, 0]) + margin)
        ax.set_ylim(np.min(ee[:, 1]) - margin, np.max(ee[:, 1]) + margin)
        ax.set_zlim(np.min(ee[:, 2]) - margin, np.max(ee[:, 2]) + margin)

    def render(self, t):
        self.left_pt._offsets3d = (
            [self.left_ee[t, 0]], [self.left_ee[t, 1]], [self.left_ee[t, 2]]
        )
        self.right_pt._offsets3d = (
            [self.right_ee[t, 0]], [self.right_ee[t, 1]], [self.right_ee[t, 2]]
        )
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        h, w = img.shape[:2]
        if (h, w) != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return img


def render_video(images, qpos, factors, chunk_size, output_path, fps=50):
    n_frames, h, w, c = images.shape

    overlay_w, overlay_h = 240, 140
    tcp_w, tcp_h = 320, 160

    factor_ovl = FactorCurveOverlay(factors, chunk_size, width=overlay_w, height=overlay_h)
    tcp_ovl = TCP3DOverlay(compute_tcp_trajectory(qpos), factors, chunk_size, width=tcp_w, height=tcp_h)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for t in tqdm(range(n_frames), desc="Rendering video"):
        frame = images[t].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        factor_img = factor_ovl.render(t)
        frame[5:5 + overlay_h, w - overlay_w - 5:w - 5] = factor_img

        tcp_img = tcp_ovl.render(t)
        y0 = h - tcp_h - 5
        x0 = 5
        frame[y0:y0 + tcp_h, x0:x0 + tcp_w] = tcp_img

        out.write(frame)

    out.release()
    plt.close(factor_ovl.fig)
    plt.close(tcp_ovl.fig)
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render HDF5 episode to video with factor + 3D TCP overlay")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 episode file")
    parser.add_argument("--output_dir", type=str, default="./videos", help="Output directory")
    parser.add_argument("--camera", type=str, default="top", help="Camera name (e.g. top, left_wrist, right_wrist)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size (must match compute_factors.py)")
    parser.add_argument("--dtw_threshold", type=float, required=True, help="DTW threshold (must match compute_factors.py)")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.hdf5_path}")
    with h5py.File(args.hdf5_path, "r") as f:
        images = f[f"/observations/images/{args.camera}"][()]
        qpos = f["/observations/qpos"][()]
        n_frames = images.shape[0]

    factors = load_factors_hdf5(args.hdf5_path, args.chunk_size, args.dtw_threshold)
    if factors is None:
        threshold_str = str(args.dtw_threshold).replace(".", "_")
        print(f"WARNING: adaptive_factors_{args.chunk_size}_{threshold_str} not found in HDF5, using all 1.0")
        factors = np.ones(n_frames, dtype=np.float64)

    base_name = os.path.splitext(os.path.basename(args.hdf5_path))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}_video.mp4")

    render_video(images, qpos, factors, args.chunk_size, output_path, args.fps)


if __name__ == "__main__":
    main()
