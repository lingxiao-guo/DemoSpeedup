#!/usr/bin/env python3
"""
自适应加速因子训练与评估主流程。

需先运行 compute_factors.py 生成 /adaptive_factors 到 HDF5 文件。

用法:
  # 训练
  python adaptive_acc/train.py \
      --task_name sim_insertion_human --ckpt_dir ... \
      --policy_class ACT --chunk_size 50 --speedup \
      --num_epochs 16000 --kl_weight 10 --batch_size 8 \
      --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0

  # 评估
  python adaptive_acc/train.py \
      --task_name sim_insertion_human --ckpt_dir ... \
      --policy_class ACT --chunk_size 50 --speedup --eval \
      --num_epochs 0 --kl_weight 10 --batch_size 8 \
      --hidden_dim 512 --dim_feedforward 3200 --lr 1e-5 --seed 0
"""

import torch
import numpy as np
import os
import sys
import h5py
import argparse
import pickle
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from act.imitate_episodes import (
    train_bc,
    eval_bc,
    set_seed,
)
from act.act_utils import (
    EpisodicDataset,
    get_norm_stats,
)
from adaptive_acc.adaptive_utils import (
    process_action_adaptive,
    load_factors_hdf5,
)


class AdaptiveEpisodicDataset(EpisodicDataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        chunk_size,
        dtw_threshold=None,
        speedup=False,
        constant_waypoint=None,
        policy_class="ACT",
    ):
        self.chunk_size = chunk_size
        self.dtw_threshold = dtw_threshold
        self.speedup = speedup
        super().__init__(
            episode_ids,
            dataset_dir,
            camera_names,
            norm_stats,
            speedup=speedup,
            constant_waypoint=constant_waypoint,
            policy_class=policy_class,
        )

    def __getitem__(self, index):
        sample_full_episode = False
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

            factors_read = False
            if self.speedup and self.dtw_threshold is not None:
                all_factors = load_factors_hdf5(dataset_path, self.chunk_size, self.dtw_threshold)
                if all_factors is not None:
                    factor = all_factors[start_ts]
                    factors_read = True
                else:
                    import warnings
                    warnings.warn(f"speedup=True but adaptive_factors_{self.chunk_size}_{str(self.dtw_threshold).replace('.', '_')} not found in {dataset_path}. "
                                  f"Run compute_factors.py first. Falling back to no speedup.")

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.speedup and factors_read:
            action_data = process_action_adaptive(action_data, factor, self.chunk_size, action_len)

        return image_data, qpos_data, action_data, is_pad


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
              chunk_size, dtw_threshold=None, speedup=False, constant_waypoint=None, policy_class="ACT"):
    print(f"\nData from: {dataset_dir}")
    print(f"Speedup: {speedup}, chunk_size={chunk_size}, dtw_threshold={dtw_threshold}\n")

    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    train_dataset = AdaptiveEpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, chunk_size,
        dtw_threshold=dtw_threshold, speedup=speedup, constant_waypoint=constant_waypoint, policy_class=policy_class,
    )
    val_dataset = AdaptiveEpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, chunk_size,
        dtw_threshold=dtw_threshold, speedup=speedup, constant_waypoint=constant_waypoint, policy_class=policy_class,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                  pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True,
                                pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="ckpt_dir")
    parser.add_argument("--policy_class", type=str, required=True, help="policy_class (ACT or DP)")
    parser.add_argument("--task_name", type=str, required=True, help="task_name")
    parser.add_argument("--batch_size", type=int, required=True, help="batch_size")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument("--num_epochs", type=int, required=True, help="num_epochs")
    parser.add_argument("--lr", type=float, required=True, help="lr")
    parser.add_argument("--kl_weight", type=int, required=False, help="KL Weight")
    parser.add_argument("--chunk_size", type=int, required=False, help="chunk_size")
    parser.add_argument("--hidden_dim", type=int, required=False, help="hidden_dim")
    parser.add_argument("--dim_feedforward", type=int, required=False, help="dim_feedforward")
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--constant_waypoint", type=int, default=None, help="constant_waypoint")
    parser.add_argument("--diffusion_policy_cfg", type=str, default="act/image_aloha_diffusion_policy_cnn.yaml")
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument("--eval_speed", action="store_true")
    parser.add_argument("--dtw_threshold", type=float, default=None, help="DTW threshold for loading factors (required when --speedup)")

    args = parser.parse_args()

    set_seed(1)
    is_eval = args.eval
    ckpt_dir = args.ckpt_dir
    policy_class = args.policy_class
    onscreen_render = args.onscreen_render
    task_name = args.task_name
    batch_size_train = args.batch_size
    batch_size_val = args.batch_size
    num_epochs = args.num_epochs
    speedup = args.speedup
    constant_waypoint = args.constant_waypoint
    temporal_agg = args.temporal_agg

    is_sim = True
    if is_sim:
        from act.constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"

    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args.lr, "num_queries": args.chunk_size, "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim, "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone, "backbone": backbone,
            "enc_layers": enc_layers, "dec_layers": dec_layers, "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args.lr, "lr_backbone": lr_backbone, "backbone": backbone,
            "num_queries": 1, "camera_names": camera_names,
        }
    elif policy_class == "DP":
        encoder_config = {
            "lr": args.lr, "lr_backbone": lr_backbone, "backbone": backbone,
            "num_queries": 1, "camera_names": camera_names,
        }
        policy_config = {
            "cfg": args.diffusion_policy_cfg, "encoder": encoder_config, "num_queries": 24,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs, "ckpt_dir": ckpt_dir, "episode_len": episode_len,
        "state_dim": state_dim, "lr": args.lr, "policy_class": policy_class,
        "onscreen_render": onscreen_render, "policy_config": policy_config,
        "task_name": task_name, "seed": args.seed, "speedup": speedup,
        "temporal_agg": temporal_agg, "camera_names": camera_names,
        "real_robot": not is_sim, "dataset_path": dataset_dir,
    }

    if is_eval:
        ckpt_names = ["policy_last.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
        chunk_size=args.chunk_size, dtw_threshold=args.dtw_threshold, speedup=speedup,
        constant_waypoint=constant_waypoint, policy_class=policy_class,
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


if __name__ == "__main__":
    main()
