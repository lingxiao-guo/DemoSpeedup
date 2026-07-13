# Adaptive Acceleration Factors

基于 DTW 轨迹分析的**每帧自适应加速因子**方案，替代 DemoSpeedup 原有的二元熵标注流程。共 3 个脚本 + 1 个工具模块。

## 原理

通过 DTW 比较 **reference chunk（chunk_size）与 extended chunk（L=chunk_size×factor）** 的相似度，找到每帧可承受的最大加速因子。训练时按因子对 action chunk 做自适应重采样。

```
原始: frame t 训练 target = actions[t : t+chunk_size]
加速: frame t 训练 target = resample(actions[t : t+chunk_size×factor[t]], chunk_size)
```

## 项目结构

```
adaptive_acc/
├── adaptive_utils.py        # 共享工具（extract_action_chunk, resample_linear, HDF5 因子读写）
├── compute_factors.py       # 脚本1: DTW 分析 → adaptive_factors_{chunk_size}_{threshold} (HDF5)
├── train.py                 # 脚本2: 训练 + --eval 评估
├── visualize.py             # 脚本3: 渲染视频（camera + 3D TCP + 因子曲线 overlay）
└── README.md                # 本文件
```

## 环境依赖

```bash
conda activate aloha
pip install dtaidistance
```

其余依赖 (`numpy`, `scipy`, `h5py`, `modern_robotics`, `torch`) 已包含在 aloha 环境中。

## 脚本1: compute_factors.py — 计算因子

对每条 episode 逐帧计算最优加速因子，直接写入源 HDF5 的 `adaptive_factors_{chunk_size}_{threshold}`。

```bash
# 默认: 14D action DTW
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human \
    --chunk_size 50 \
    --dtw_threshold 0.03

# 使用 6D TCP (FK from qpos)
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human \
    --chunk_size 50 \
    --dtw_threshold 0.03 \
    --tcp_only
```

**参数**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--task_name` | 必填 | 任务名，对应 `data/` 下数据集目录 |
| `--chunk_size` | 必填 | action chunk 长度，与训练一致 |
| `--dtw_threshold` | 必填 | DTW 阈值，因子=DTW<阈值 的最大 factor |
| `--tcp_only` | False | FK 转 6D TCP，否则用 14D action |
| `--max_factor` | 3.0 | 最大加速因子 |
| `--scale_step` | 0.1 | 因子搜索步长 |
| `--frame_stride` | 1 | 帧采样间隔（>1 时线性插值填充） |
| `--dtw_window` | 5 | DTW Sakoe-Chiba 窗口（0=无约束） |
| `--filter_window` | 9 | 中值滤波窗口 |
| `--episodes 0 1 2` | 全部 | 指定 episode |
| `--num_workers` | 0 | 并行数（0=自动） |

**末尾帧处理**: 剩余帧 ≤ chunk_size → 返回 inf DTW，factor 退回 1.0；剩余帧 > chunk_size 但不足完整 L 时自动截断，DTW 诚实反映加速可行性。

**输出**: 修改 `data/{task_name}/episode_*.hdf5`，写入 `adaptive_factors_{chunk_size}_{threshold}` (例: `adaptive_factors_50_0_03`)。同一 HDF5 可共存多组不同参数计算的因子。终端打印每 episode 统计摘要。

## 脚本2: train.py — 训练 / 评估

训练时读取 HDF5 中的 `adaptive_factors_{chunk_size}_{threshold}`，按当前帧因子构建训练样本。通过 `--eval` 切换评估模式。

```bash
# 训练 (--dtw_threshold 和 --chunk_size 须与 compute_factors 一致)
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --dtw_threshold 0.03 --speedup --temporal_agg \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0

# 评估
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --num_epochs 0 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0 --eval

# DP 策略
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/DP/sim_insertion_human/ \
    --policy_class DP --chunk_size 48 --dtw_threshold 0.03 --speedup --temporal_agg \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0
```

**参数**:

| 参数 | 说明 |
|------|------|
| `--eval` | 评估模式（不训练） |
| `--speedup` | 启用自适应因子压缩 |
| `--dtw_threshold` | speedup 时必填，用于定位 `adaptive_factors_{chunk_size}_{threshold}` |
| `--temporal_agg` | 时序聚合 |
| 其余 | 与原 `imitate_episodes.py` 一致 |

## 脚本3: visualize.py — 渲染视频

将 HDF5 episode 渲染为 MP4 视频，右上角叠加因子曲线指示器，左下角叠加 3D TCP 轨迹。

```
┌───────────────────────────────────────┐
│                      ┌──────────────┐ │
│                      │  因子曲线     │ │
│                      │  ● 当前帧     │ │
│                      │  ◆ 链式节点  │ │
│          Camera View   └──────────────┘ │
│  (主体)                                 │
│                                         │
│ ┌──────────┐                            │
│ │ Left 3D  │                            │
│ │ Right 3D │                            │
│ │ ● 当前点  │                            │
│ └──────────┘                            │
└───────────────────────────────────────┘
```

额外依赖: `opencv-python` (cv2)

```bash
# chunk_size 和 dtw_threshold 须与 compute_factors 一致
python adaptive_acc/visualize.py \
    --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
    --output_dir ./videos \
    --camera top --chunk_size 50 --dtw_threshold 0.03 --fps 50

# 如 HDF5 中没有匹配的字段，则显示全 1.0 曲线
python adaptive_acc/visualize.py \
    --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
    --output_dir ./videos \
    --camera top --chunk_size 50 --dtw_threshold 0.03 --fps 50
```

**参数**:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--hdf5_path` | 必填 | HDF5 episode 文件路径 |
| `--output_dir` | `./videos` | 输出目录 |
| `--camera` | `top` | 相机名（top/left_wrist/right_wrist） |
| `--chunk_size` | 50 | 需与 compute_factors 一致 |
| `--dtw_threshold` | 必填 | 需与 compute_factors 一致 |
| `--fps` | 50 | 视频帧率 |

**输出**: `{output_dir}/{episode_name}_video.mp4`

## 典型命令速查

```bash
# === sim_insertion_human ===

# 1. 计算因子
python adaptive_acc/compute_factors.py \
    --task_name sim_insertion_human --chunk_size 50 --dtw_threshold 0.03

# 2. 训练
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --dtw_threshold 0.03 --speedup --temporal_agg \
    --num_epochs 16000 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0

# 3. 评估
python adaptive_acc/train.py \
    --task_name sim_insertion_human \
    --ckpt_dir data/outputs/adaptive/ACT/sim_insertion_human/ \
    --policy_class ACT --chunk_size 50 --speedup --temporal_agg \
    --num_epochs 0 --kl_weight 10 --batch_size 8 \
    --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0 --eval

# 4. 可视化
python adaptive_acc/visualize.py \
    --hdf5_path data/sim_insertion_human/episode_0.hdf5 \
    --camera top --chunk_size 50 --dtw_threshold 0.03
```
