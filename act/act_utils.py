import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
import IPython
import cv2

e = IPython.embed


def relabel_waypoints(arr, waypoint_indices):
    start_idx = 0
    for key_idx in waypoint_indices:
        # Replace the items between the start index and the key index with the key item
        arr[start_idx:key_idx] = arr[key_idx]
        start_idx = key_idx
    return arr

def put_text(img, text, is_waypoint=False, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img

def bottom_20_percent_value(lst):
    # 对列表进行排序
    lst = list(lst)
    sorted_lst = sorted(lst)
    
    # 计算前80%的位置索引
    bottom_20_index = int(len(sorted_lst)*0.2) # 1: 80% 0.8 0.5 0.3
    
    # 检查元素是否在后20%的范围内
    return sorted_lst[bottom_20_index]

def plot_3d_trajectory(ax, traj_list, actions_var_norm=None, distance=None, label=None, gripper=None, legend=True, add=None):
    """Plot a 3D trajectory."""
    mark = None
    if actions_var_norm is not None:
        import math
        # actions_var_log = [math.log(var+1e-8,1.5) for var in actions_var_norm] # math.log(var+1e-8)
        # actions_var_log = np.array(actions_var_log)
        actions_var_log = actions_var_norm
        print(f"min log:{np.min(actions_var_log[:250])}|max log:{np.max(actions_var_log[:250])}")
        # mark = (actions_var_log +14)/(14-2)
        # for insertion:
        # mark = (actions_var_log +20)/(18)
        mark = np.array(actions_var_log)
        # mark = np.exp(mark)/np.sum(np.exp(mark))
        key = bottom_20_percent_value(mark[:220])
        print(f"20% idx:{key}")
        # mark = (actions_var_norm)/5
        # mark = (actions_var_log - np.mean(actions_var_log))/np.var(actions_var_log)
        # mark = np.clip(1-actions_var_norm, 0,1)
    elif distance is not None:
        mark = [d*50 for d in distance]

    l = label
    num_frames = len(traj_list)
    count = 0
    for i in range(num_frames):
        # change the color if the gripper state changes
        gripper_state_changed = (
            gripper is not None and i > 0 and gripper[i] != gripper[i - 1]
        )
        if label == "pred" or label == "waypoints":
            if mark is None:
                if gripper_state_changed or (add is not None and i in add):
                    c = mpl.cm.Oranges(0.2 + 0.5 * i / num_frames)
                else:
                    # c = mpl.cm.Reds(0.5 + 0.5 * i / num_frames)
                    c = mpl.cm.Reds(0.1)
            else:
                c = mpl.cm.Reds(np.clip((0.5-mark[i]),0,1))
        elif label == "gt" or label == "ground truth" or label == "demos replay":
            if mark is None:
                if gripper_state_changed:
                    c = mpl.cm.Greens(0.2 + 0.5 * i / num_frames)
                else:
                    c = mpl.cm.Blues(0.9 + 0.1 * i / num_frames)
            else:
                c = mpl.cm.Blues(0.5+0.5*np.clip((0.5-mark[i]),0,1))
        else:
                # c = mpl.cm.Greens(0.5 + 0.5 * i / num_frames)
                if mark[i] < 0.8 and i>20 and i<240:  # 0.4 for insertion, 0.3 for transfer
                    c = mpl.cm.Blues(np.clip((1-mark[i]),0,1))
                    count+=1
                else:
                    c = mpl.cm.Reds(np.clip((1-mark[i]),0,1))

        # change the marker if the gripper state changes
        if gripper_state_changed:
            if gripper[i] == 1:  # open
                marker = "D"
            else:  # close
                marker = "s"
        else:
            marker = "o"

        # plot the vector between the current and the previous point
        if (label == "pred" or label == "action" or label == "waypoints") and i > 0:
            v = traj_list[i] - traj_list[i - 1]
            ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color="r",
                alpha=0.5,
                # linewidth=3,
            )

        # if label is waypoint, make the marker D, and slightly bigger
        if add is not None and i in add:
            marker = "D"
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=10,
            )
        else:
            if i > 0:
                v = traj_list[i] - traj_list[i - 1]
                ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color=c,
                alpha=0.5,
                # linewidth=3,
            )
            if gripper_state_changed:
                if gripper[i] == 1:  # open
                   marker = "D"
                else:  # close
                   marker = "s"
                ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=5,
                )
        l = None

    if legend:
        ax.legend()
    print("cautious rates:",count/250)
    # ax.w_xaxis.set_pane_color((173/255, 216/255, 230/255, 1.0))
    # ax.w_yaxis.set_pane_color((173/255, 216/255, 230/255, 1.0))
    # ax.w_zaxis.set_pane_color((173/255, 216/255, 230/255, 1.0))


def mark_3d_trajectory(ax, traj_list, mark_list, key_mark=None, label=None, gripper=None, legend=True, add=None):
    """Plot a 3D trajectory."""
    mark = None
    l = label
    num_frames = len(traj_list)
    for i in range(num_frames):
        # change the color if the gripper state changes
        gripper_state_changed = (
            gripper is not None and i > 0 and gripper[i] != gripper[i - 1]
        )
        if label == "pred" or label == "waypoints":
                if gripper_state_changed or (add is not None and i in add):
                    c = mpl.cm.Oranges(0.2 + 0.5 * i / num_frames)
                else:
                    # c = mpl.cm.Reds(0.5 + 0.5 * i / num_frames)
                    c = mpl.cm.Reds(0.3)
        else: 
                if gripper_state_changed:
                    c = mpl.cm.Greens(0.2 + 0.5 * i / num_frames)
                else:
                    c = mpl.cm.Blues(0.5 + 0.5 * int(mark_list[i]))

        # change the marker if the gripper state changes
        if gripper_state_changed:
            if gripper[i] == 1:  # open
                marker = "D"
            else:  # close
                marker = "s"
        else:
            marker = "o"

        # plot the vector between the current and the previous point
        if (label == "pred" or label == "action" or label == "waypoints") and i > 0:
            v = traj_list[i] - traj_list[i - 1]
            ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color="r",
                alpha=0.5,
                # linewidth=3,
            )

        # if label is waypoint, make the marker D, and slightly bigger
        if i>0:
            if mark_list[i]==True and mark_list[i-1]==False:
                marker = "*"
                ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color="y",
                markersize=10,
                )
            else:
                v = traj_list[i] - traj_list[i - 1]
                ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color=c,
                alpha=0.5,
                # linewidth=3,
            )

        
        l = None

    if legend:
        ax.legend()

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=False,
        constant_waypoint=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_waypoint = use_waypoint
        self.constant_waypoint = constant_waypoint
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = True #root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

            if self.use_waypoint and self.constant_waypoint is None:
                waypoints = root["/waypoints"][()]

        if self.use_waypoint:
            # constant waypoints
            if self.constant_waypoint is not None:
                assert self.constant_waypoint > 0
                waypoints = np.arange(1, action_len, self.constant_waypoint)
                if len(waypoints) == 0:
                    waypoints = np.array([action_len - 1])
                elif waypoints[-1] != action_len - 1:
                    waypoints = np.append(waypoints, action_len - 1)
            # auto waypoints
            else:
                waypoints = waypoints - start_ts
                waypoints = waypoints[waypoints >= 0]
                waypoints = waypoints[waypoints < action_len]
                waypoints = np.append(waypoints, action_len - 1)
                waypoints = np.unique(waypoints)
                waypoints = waypoints.astype(np.int32)

            action = relabel_waypoints(action, waypoints)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    use_waypoint=False,
    constant_waypoint=None,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=use_waypoint,
        constant_waypoint=constant_waypoint,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        use_waypoint=use_waypoint,
        constant_waypoint=constant_waypoint,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)