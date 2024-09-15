import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import pickle
import torch

from waypoint_extraction.extract_waypoints import optimize_waypoint_selection,dp_waypoint_selection, greedy_waypoint_selection, entropy_waypoint_selection,gripper_change_detect
from act.policy import ACTPolicy, CNNMLPPolicy
from act.visualize_episodes import save_videos
from act.act_utils import set_seed
from act.constants import DT

def main(args):
    num_waypoints = []
    num_frames = []
    
    ckpt_dir = args.ckpt_dir
    policy_class = args.policy_class
    task_name = args.task_name
    state_dim = 14
    lr_backbone = 1e-5
    camera_names = "top"
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args.lr,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }

    config = {
        "ckpt_dir": ckpt_dir,
        "state_dim": state_dim,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args.seed,
        "temporal_agg": args.temporal_agg,
        "camera_names": camera_names
    }
    ckpt_names = [f"policy_last.ckpt"]
    num_rollouts = args.end_idx-args.start_idx+1
    """if not args.plot_3d:
        for ckpt_name in ckpt_names:
            entropy_list = label_entropy(args.dataset, num_rollouts, config, ckpt_name)
            print(f"entropy list:{entropy_list[0].shape}")"""
    
    # load data
    for i in tqdm(range(args.start_idx, args.end_idx + 1)):
        dataset_path = os.path.join(args.dataset, f"episode_{i}.hdf5")
        with h5py.File(dataset_path, "r+") as root:
            qpos = root["/observations/qpos"][()]
            images = root["/observations/images/top"][()]
            entropy = root["/entropy"][()]
            variance = root["/variance"][()]
            waypoints = root["/waypoints"][()]
            entropy = np.array(entropy)
            variance = np.array(variance)[:,None]
            if args.use_ee:
                qpos = np.array(qpos)  # ts, dim

                # calculate EE pose
                from act.convert_ee import get_ee

                left_arm_ee = get_ee(qpos[:, :6], qpos[:, 6:7])
                right_arm_ee = get_ee(qpos[:, 7:13], qpos[:, 13:14])
                qpos = np.concatenate([left_arm_ee, right_arm_ee], axis=1)
            """
            # select waypoints            
            waypoints, distance = dp_waypoint_selection( # if it's too slow, use greedy_waypoint_selection
                env=None,
                actions=qpos,
                gt_states=qpos,
                err_threshold=args.err_threshold,
                pos_only=True,
            )
            
            waypoints, distance = entropy_waypoint_selection(
                env=None,
                actions=qpos,
                gt_states=qpos,
                actions_entropy=entropy,
                err_threshold=args.err_threshold,
                pos_only=True,
            )
            """
            waypoints, distance = optimize_waypoint_selection( # if it's too slow, use greedy_waypoint_selection
                env=None,
                actions=qpos,
                gt_states=qpos,
                err_threshold=args.err_threshold,
                pos_only=True,
                entropy=entropy,
            )
            
            print(
                f"Episode {i}: {len(qpos)} frames -> {len(waypoints)} waypoints (ratio: {len(qpos)/len(waypoints):.2f})"
            )
            num_waypoints.append(len(waypoints))
            num_frames.append(len(qpos))

            # save waypoints
            if args.save_waypoints:
                name = f"/waypoints"  # /entropy_waypoints
                try:
                    root[name] = waypoints
                except:
                    # if the waypoints dataset already exists, ask the user if they want to overwrite
                    # print("waypoints dataset already exists. Overwrite? (y/n)")
                    # ans = input()
                    ans = "y"
                    if ans == "y":
                        del root[name]
                        root[name] = waypoints
            
            gripper_indices = gripper_change_detect(qpos, qpos)
            entropy[gripper_indices] = np.min(entropy)*0.99
            mean = np.mean(entropy)
            entropy[np.where(entropy>mean)[0]] = 5
            entropy[np.where(entropy<mean)[0]] = 0.1
            # visualize ground truth qpos and waypoints
            if args.plot_3d:
                if not args.use_ee:
                    qpos = np.array(qpos)  # ts, dim
                    from act.convert_ee import get_xyz

                    left_arm_xyz = get_xyz(qpos[:, :6])
                    right_arm_xyz = get_xyz(qpos[:, 7:13])
                else:
                    left_arm_xyz = left_arm_ee[:, :3]
                    right_arm_xyz = right_arm_ee[:, :3]

                # Find global min and max for each axis
                all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
                min_x, min_y, min_z = np.min(all_data, axis=0)
                max_x, max_y, max_z = np.max(all_data, axis=0)

                fig = plt.figure(figsize=(20, 10))
                ax1 = fig.add_subplot(121, projection="3d") 
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("z")
                ax1.set_title("Left", fontsize=20)
                ax1.set_xlim([min_x, max_x])
                ax1.set_ylim([min_y, max_y])
                ax1.set_zlim([min_z, max_z])
                
                # from utils.utils import plot_3d_trajectory
                from act.act_utils import plot_3d_trajectory
                
                ax2 = fig.add_subplot(122, projection="3d")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_zlabel("z")
                ax2.set_title("Right", fontsize=20)
                ax2.set_xlim([min_x, max_x])
                ax2.set_ylim([min_y, max_y])
                ax2.set_zlim([min_z, max_z])

                # prepend 0 to waypoints to include the initial state
                waypoints = [0] + waypoints
                plot_3d_trajectory(
                    ax1,
                    [left_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for left_arm_xyz
                plot_3d_trajectory(
                    ax2,
                    [right_arm_xyz[i] for i in waypoints],
                    label="waypoints",
                    legend=False,
                )  # Plot waypoints for right_arm_xyz"""
                plot_3d_trajectory(ax1, left_arm_xyz, label="gt", legend=False)
                plot_3d_trajectory(ax2, right_arm_xyz, label="gt", legend=False)
                
                # plot_3d_trajectory(ax1, left_arm_xyz, distance=distance, label="ground truth", legend=False)
                # plot_3d_trajectory(ax2, right_arm_xyz, distance=distance,  label="ground truth", legend=False)                
                fig.suptitle(f"Task: {args.dataset.split('/')[-1]}", fontsize=30) 

                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)

                fig.savefig(
                    f"plot/act/{args.dataset.split('/')[-1]}_{i}_t_{args.err_threshold}_waypoints.png"
                )
                plt.close(fig)

            root.close()

    print(
        f"Average number of waypoints: {np.mean(num_waypoints)} \tAverage number of frames: {np.mean(num_frames)} \tratio: {np.mean(num_frames)/np.mean(num_waypoints)}"
    )


def label_entropy(dataset, num_rollouts, config, ckpt_name, save_episode=True, save_labels=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    variance_step = 1
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    # load environment
    from act.act_utils import put_text, plot_3d_trajectory
    # Utilize policy to label every step in the trajectory
    query_frequency = 1
    num_queries = policy_config["num_queries"]
    
    entropy_list = []
    # TODO: change the entropy label way to parallel rollout labelling
    for rollout_id in tqdm(range(num_rollouts)):
        dataset_path = os.path.join(dataset, f"episode_{rollout_id}.hdf5")
        with h5py.File(dataset_path, "r+") as root:
            all_qpos = root["/observations/qpos"][()]
            images = root["/observations/images/top"][()]
        
        max_timesteps = int(np.array(all_qpos).shape[0])
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_var = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 1]
            ).cuda()
            all_time_marginal_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        actions_var = []
        actions_marginal_var = []
        traj_action_entropy = []
        traj_marginal_entropy = []
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                
                ### process previous timestep to get qpos and image_list
                qpos_numpy = np.array(all_qpos[t])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                obs = {"images":{}}
                for cam_name in camera_names:
                    obs["images"][cam_name] = images[t]
                curr_image = process_image(obs["images"], camera_names)
             
                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions, action_entropy, action_marginal_entropy = policy.get_entropy(qpos, curr_image)
                        # TODO: sample multiple z and estimate variance/entropy, draw it in the video
                        # num_samples = 2
                        # action_samples = []
                        # for n in range(num_samples):
                        #     all_actions = policy(qpos, curr_image)
                        #     action_samples.append(all_actions)
                        # action_samples = torch.stack(action_samples)
                        action_samples = all_actions
                        all_actions = torch.mean(action_samples, dim=0)
                        # left_actions_var = torch.var(action_samples[:,:,:variance_step,:].reshape(num_samples,-1), dim=0)
                        # right_actions_var = torch.var(action_samples[:,:,:variance_step,:].reshape(num_samples,-1), dim=0)
                        action_var = torch.var(action_samples, dim=0)
                        # Only calculate the first step variance
                        
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        all_time_var[[t], t : t + num_queries] = action_var
                        all_time_entropy[[t], t : t + num_queries] = action_entropy
                        all_time_marginal_entropy[[t], t : t + num_queries] = action_marginal_entropy
                        actions_for_curr_step = all_time_actions[:, t]
                        action_var_curr_step = all_time_var[:, t]
                        action_entropy_curr_step = all_time_entropy[:, t]
                        action_marginal_entropy_curr_step = all_time_marginal_entropy[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        action_var_curr_step = action_var_curr_step[actions_populated]
                        action_entropy_curr_step = action_entropy_curr_step[actions_populated]
                        action_marginal_entropy_curr_step = action_marginal_entropy_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                        action_var = (action_var_curr_step * exp_weights).sum(
                            dim=0
                        )
                        action_entropy = (action_entropy_curr_step * exp_weights).sum(
                            dim=0
                        )
                        action_marginal_entropy = (action_marginal_entropy_curr_step *exp_weights).sum(
                            dim=0
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]

                    # actions_var.append(torch.cat((torch.mean(left_actions_var,dim=-1,keepdim=True),torch.mean(right_actions_var,dim=-1,keepdim=True)),dim=-1))
                    actions_var.append(torch.mean(action_var, dim=-1))
                    actions_marginal_var.append(action_var)
                    traj_action_entropy.append(action_entropy)
                    traj_marginal_entropy.append(action_marginal_entropy)
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                    
                ### store processed image for video 
                entropy_numpy = np.array(traj_action_entropy[-1].cpu())
                store_imgs = {}
                for key, img in obs["images"].items():
                    store_imgs[key] = put_text(img,entropy_numpy)
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
   
            plt.close()

        # draw trajectory curves
        actions_var = torch.stack(actions_var)
        actions_var = np.array(actions_var.cpu())
        actions_marginal_var = torch.stack(actions_marginal_var)
        actions_marginal_var = np.array(actions_marginal_var.cpu())
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())
        traj_marginal_entropy = torch.stack(traj_marginal_entropy)
        traj_marginal_entropy = np.array(traj_marginal_entropy.cpu())
        
        max_var = np.max(actions_var,axis=0)
        min_var = np.min(actions_var,axis=0)
        actions_var_norm = (actions_var - min_var) / (max_var - min_var)
        max_entropy = np.max(traj_action_entropy, axis=0)
        min_entropy = np.min(traj_action_entropy, axis=0)
        actions_entropy_norm = (traj_action_entropy-min_entropy)/(max_entropy-min_entropy)
        
        # Don't to normalization to marginal entropy, 
        # Since the normalization operation has already done in processing the dataset,
        # AND the abs value of entropy could mean sth. Normalization may hurt this.
        marginal_entropy_norm = traj_marginal_entropy
        actions_marginal_var_norm = actions_marginal_var

        entropy_list.append(actions_entropy_norm)

        qpos = np.array(qpos_list)  # ts, dim
        from act.convert_ee import get_xyz

        left_arm_xyz = get_xyz(qpos[:, :6])
        right_arm_xyz = get_xyz(qpos[:, 7:13])
        # Find global min and max for each axis
        all_data = np.concatenate([left_arm_xyz, right_arm_xyz], axis=0)
        min_x, min_y, min_z = np.min(all_data, axis=0)
        max_x, max_y, max_z = np.max(all_data, axis=0)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection="3d") 
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("Left", fontsize=20)
        ax1.set_xlim([min_x, max_x])
        ax1.set_ylim([min_y, max_y])
        ax1.set_zlim([min_z, max_z])
        plot_3d_trajectory(ax1, left_arm_xyz, actions_var_norm,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, actions_var_norm,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)

        # Only save successful video/plot
        if save_episode : 
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/video{rollout_id}_{variance_step}.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/{task_name}_{rollout_id}_{variance_step}_waypoints.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/{task_name}_{rollout_id}_{variance_step}_view.png")
            )
        if save_labels:
            with h5py.File(dataset_path, "r+") as root:
                name = f"/entropy"
                try:
                    root[name] = actions_entropy_norm
                except:
                    del root[name]
                    root[name] = actions_entropy_norm  
                name = f"/variance"
                try:
                    root[name] = actions_var_norm[:,None]
                except:
                    del root[name]
                    root[name] = actions_var_norm[:,None]
                name = f"/marginal_entropy"
                try:
                    root[name] = actions_marginal_var_norm
                except:
                    del root[name]
                    root[name] = actions_marginal_var_norm
        plt.close(fig)
        root.close()
        
    # save select points to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        # TODO: f.write selected waypoints and sped up ?X 
        f.write("\n\n")
    
    return entropy_list
        
def process_image(images, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/act/sim_transfer_cube_scripted",
        # default="data/act/sim_insertion_scripted",
        # default="data/act/sim_transfer_cube_human",
        # default="data/act/sim_insertion_human",
        # default="data/act/aloha_screw_driver",
        # default="data/act/aloha_coffee",
        # default="data/act/aloha_towel",
        # default="data/act/aloha_coffee_new",
        help="path to hdf5 dataset",
    )

    # index of the trajectory to playback. If omitted, playback trajectory 0.
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="(optional) start index of the trajectory to playback",
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=49,
        help="(optional) end index of the trajectory to playback",
    )

    # error threshold for reconstructing the trajectory
    parser.add_argument(
        "--err_threshold",
        type=float,
        default=0.05,
        help="(optional) error threshold for reconstructing the trajectory",
    )

    # whether to save waypoints
    parser.add_argument(
        "--save_waypoints",
        action="store_true",
        help="(optional) whether to save waypoints",
    )

    # whether to use the ee space for waypoint selection
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="(optional) whether to use the ee space for waypoint selection",
    )
    
    parser.add_argument("--seed",  type=int, default=0,help="seed")
    
    # whether to plot 3d
    parser.add_argument(
        "--plot_3d",
        action="store_true",
        help="(optional) whether to plot 3d",
    )

    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        default="ACT",
        type=str,
        help="policy_class, capitalize",
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )

    parser.add_argument("--lr", default=1e-5, type=float, help="lr")

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", default=10, type=int, help="KL Weight"
    )
    parser.add_argument(
        "--chunk_size", action="store", default=50, type=int, help="chunk_size"
    )
    parser.add_argument(
        "--hidden_dim", action="store", default=512, type=int, help="hidden_dim"
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        default = 3200
    )
    parser.add_argument("--temporal_agg", default=True, action="store_true")
    
    args = parser.parse_args()
    main(args)
