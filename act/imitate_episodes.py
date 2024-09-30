import torch
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import pickle
import h5py
import math
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import wandb
from act.constants import DT
from act.constants import PUPPET_GRIPPER_JOINT_OPEN
from act.act_utils import load_data  # data functions
from act.act_utils import sample_box_pose, sample_insertion_pose  # robot functions
from act.act_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy
from act.detr.models.entropy_utils import KDE
from act.visualize_episodes import save_videos
from act.sim_env import BOX_POSE,make_sim_env
import sys 
sys.path.append("..") 
from waypoint_extraction.extract_waypoints import optimize_waypoint_selection
import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    is_plot = args["plot"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    use_waypoint = args["use_waypoint"]
    constant_waypoint = args["constant_waypoint"]
    use_wandb = args["use_wandb"]
    if use_wandb:
        wandb.init(project='act-aloha', name=task_name, entity='Lingxiao-guo')
    if use_waypoint:
        print("Using waypoint")
    if constant_waypoint is not None:
        print(f"Constant waypoint: {constant_waypoint}")

    # get task parameters
    # is_sim = task_name[:4] == 'sim_'
    is_sim = True  # hardcode to True to avoid finding constants from aloha
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
    }
    dataset_path = os.path.join(dataset_dir, f"episode_0.hdf5")
    with h5py.File(dataset_path, "r") as root:
            entropy_var = root["/entropy_var"][()]
            entropy_mean = root["/entropy_mean"][()]
            entropy_min = root["/min_var"][()]
            entropy_max = root["/max_var"][()]
            entropy_var = torch.from_numpy(np.array(entropy_var)).float().cuda().unsqueeze(0)
            entropy_mean = torch.from_numpy(np.array(entropy_mean)).float().cuda().unsqueeze(0)
    H_dict = {"mean":entropy_mean,"var":entropy_var,"max":entropy_max,"min":entropy_min}
   
    if is_eval:
        ckpt_names = [f"policy_last.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            # success_rate, avg_return = plot_trajectory_variance(config, ckpt_name, args["save_demos"],save_episode=True)
            success_rate, avg_return = eval_bc(config, ckpt_name, H_dict,save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()
    
    if args["eval_speed"]:
        ckpt_names = [f"policy_last.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            # success_rate, avg_return = plot_trajectory_variance(config, ckpt_name, args["save_demos"],save_episode=True)
            success_rate, avg_return = eval_speed_bc(config, ckpt_name, H_dict,save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    if args["visualize_entropy"]:
        ckpt_names = [f"policy_last.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = plot_trajectory_variance(config, ckpt_name, args["save_demos"],save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_waypoint,
        constant_waypoint,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, stats)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

KDE = KDE()
def eval_bc(config, ckpt_name, H_dict,save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
   
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
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env
        
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    # query_frequency = 25
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    #max_timesteps = 250
    num_rollouts = 20
    episode_returns = []
    highest_rewards = []
    H_min_list = []
    H_max_list = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        
        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 1]
            ).cuda()
            all_time_samples = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 10,state_dim]
            ).cuda()
 
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        traj_action_entropy = []
        flag = False
        waypoint_count = 0
        openloop_t = 0
        last_t = 0
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # get entropy
                        action_samples,_,action_trust = policy.get_entropy(qpos, curr_image)
                        action_samples = action_samples.squeeze().permute(1,0,2) # (chunk_len, num_samples, dim)
                        # all_actions = action_samples[:,-1].reshape(all_actions.shape)
                        entropy = torch.mean(torch.var(action_samples,dim=1))
                        # entropy = (entropy-H_dict["min"])/(H_dict["max"]-H_dict["min"])
                        # all_actions = action_trust if weights >0 else all_actions
                        # get waypoints using entropy and actions
                            
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        all_time_samples[[t], t : t + num_queries] = action_samples
                        actions_for_next_step = all_time_actions[:, t+10] # t+10
                        samples_populated = torch.all(
                            actions_for_next_step != 0, axis=1
                        )
                        samples_for_curr_step = all_time_samples[:, t]
                        samples_for_curr_step = samples_for_curr_step[samples_populated]
                        
                        all_time_entropy[[t], t : t + num_queries] = entropy
                        entropy_for_curr_step = all_time_entropy[:,t]
                        entropy_for_curr_step = entropy_for_curr_step[actions_populated]
                        # entropy = torch.mean(torch.var(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                        # entropy = (entropy-H_dict["min"])/(H_dict["max"]-H_dict["min"])
                        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        entropy = (entropy_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                        traj_action_entropy.append(entropy.squeeze())
                        weights = 5*torch.clip(0.2-entropy.squeeze(),0,0.2)  # weights: 80%: 0 20% 0~1
                        # weights = 1 if weights >0 else 0
                        weights = weights.cpu().numpy()
                        k = 0.01 # *np.exp(10*weights)  # k: 0.01~0.2
                        # Or simply mask the past 10 th k
                        # print(k)

                        """if t>50 and weights >0:
                            if flag:
                                all_time_actions[0:max(t-10,0),:,:] = 0
                                # flag = False
                            k = 0.01
                            all_time_actions[[t], t : t + num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )"""
                        # else:
                        #     flag = True
                            
                        # if t>50 and weights >0:
                        #     _,actions_for_curr_step = KDE.kde_entropy(actions_for_curr_step.unsqueeze(0),k=25) # int(40-30*weights)
                            # actions_for_curr_step = actions_for_curr_step[-25:]
                          # print(actions_for_curr_step.shape,"#")
                        # k = 0.01 
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        entropy = torch.mean(torch.var(action_samples.squeeze(),dim=1),dim=-1)
                        entropy = (entropy-H_dict["min"])/(H_dict["max"]-H_dict["min"])
                        entropy = [e for e in entropy]
                        traj_action_entropy.extend(entropy)
                        # raw_action = all_actions[:, openloop_t]
                        # openloop_t += 1

                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                """if t>50 and weights>0:
                    if flag:
                        innerloop_count = 0
                        non_gripper_idx = [0,1,2,3,4,5,7,8,9,11,12]
                        while np.linalg.norm((target_qpos-np.array(ts.observation["qpos"]))[non_gripper_idx],axis=-1)>0.02 and innerloop_count<2:
                            ts = env.step(target_qpos)
                            innerloop_count += 1
                        flag = False
                else:
                    flag = True"""

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())
        print(f"max:{np.max(traj_action_entropy)} min:{np.min(traj_action_entropy)}")
        H_min_list.append(np.min(traj_action_entropy))
        H_max_list.append(np.max(traj_action_entropy))
        traj_action_entropy = (traj_action_entropy-np.min(traj_action_entropy))/(np.max(traj_action_entropy)-np.min(traj_action_entropy))
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
        from act.act_utils import plot_3d_trajectory
        plot_3d_trajectory(ax1, left_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        
        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/video{rollout_id}.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_view.png")
            )
        
        n_groups = qpos_numpy.shape[-1]
        tstep = np.linspace(0, 1, max_timesteps-1) 
        fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)

        for n, ax in enumerate(axes):
            ax.plot(tstep, np.array(qpos_list)[1:, n], label=f'real qpos {n}')
            ax.plot(tstep, np.array(target_qpos_list)[:-1, n], label=f'target qpos {n}')
            ax.set_title(f'qpos {n}')
            ax.legend()

        plt.xlabel('timestep')
        plt.ylabel('qpos')
        plt.tight_layout()
        # fig.savefig(
        #         os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_qpos.png")
        #     )
        plt.close()
        print(f"Save qpos curve to {ckpt_dir}/plot/rollout{rollout_id}_qpos.png")
        

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)
    print(f"min entropy:{np.mean(np.array(H_min_list))}|max entropy:{np.mean(np.array(H_max_list))}")
    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))
    
    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.cuda(),
        qpos_data.cuda().to(torch.float32),
        action_data.cuda().to(torch.float32),
        is_pad.cuda(),
    )
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None

def train_eval(env, policy,config, stats,ckpt_dir,epoch_id, num_rollouts=5,save_episode=True):
    set_seed(1000)    
    state_dim = config["state_dim"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    policy.eval()
    
    env_max_reward = env.task.max_reward
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    
    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    image_list = []  # for visualization
    print("Rollout...")
    for rollout_id in tqdm(range(num_rollouts)):
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        
        ts = env.reset()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
       

    if save_episode :
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video{epoch_id}.mp4"),
            )
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)
    wandb.log({"Eval/Success rate": success_rate, "Eval/Average return":avg_return})

    return success_rate, avg_return

def train_bc(train_dataloader, val_dataloader, config,stats):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    temporal_agg = config["temporal_agg"]
    task_name = config["task_name"]
    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    env = make_sim_env(task_name)

    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 1:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            loading_status = policy.load_state_dict(torch.load(ckpt_path))
            print(loading_status)
        else:
            print("Not loading checkpoint")
            latest_idx = 0
    else:
        latest_idx = 0

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(latest_idx, num_epochs)):
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))        
        
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        e = epoch - latest_idx
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        wandb.log({'Loss/Val loss': epoch_val_loss,"Loss/Train loss": epoch_train_loss})
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        

        if epoch % 1000 == 0 and epoch !=0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            train_eval(env, policy, config, stats,ckpt_dir, epoch)
            print(f"Train loss: {epoch_train_loss:.5f}")
            print(f"Val loss:   {epoch_val_loss:.5f}")
            print(summary_string)
            
    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    wandb.finish()
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")

def plot_trajectory_variance(config, ckpt_name, save_demos=False,save_episode=True):
    set_seed(2)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
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
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env
        from act.act_utils import put_text, plot_3d_trajectory

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
    
    query_frequency = policy_config["num_queries"]
    if temporal_agg: 
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    save_id = 0
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        
        ts = env.reset()

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
        rewards = []
        actions_var = []
        actions_marginal_var = []
        traj_action_entropy = []
        traj_marginal_entropy = []
        if save_demos:
            qvel_list = []
            obs_img_list = []
            states_list = []
            actions_list = []

        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)
             
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
                        action_samples = all_actions.squeeze()
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
                        k = 0.2
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
                var_numpy = np.array(actions_var[-1].cpu())
                entropy_numpy = np.array(traj_action_entropy[-1].cpu())
                store_imgs = {}
                for key, img in obs["images"].items():
                    store_imgs[key] = put_text(img,entropy_numpy)
                    if key in camera_names and save_demos:
                        obs_img_list.append(img)
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                if save_demos:
                    qvel_list.append(obs["qvel"])
                    actions_list.append(action)
                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
   
            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)

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
        
        plot_3d_trajectory(ax1, left_arm_xyz, actions_entropy_norm,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, actions_entropy_norm,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
                
        # Only save successful video/plot
        if save_episode : #and episode_highest_reward==env_max_reward: 
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/rollout{rollout_id}.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_demos{save_id}_view.png")
            )
            
        plt.close(fig)
        
        # Saving successful demonstrations
        if save_demos and episode_highest_reward==env_max_reward: 
            dataset = {}
            dataset["/action"] = np.array(actions_list)
            dataset["/observations/images/top"] = np.array(obs_img_list)
            dataset["/observations/qpos"] = np.array(qpos_list)
            dataset["/observations/qvel"] = np.array(qvel_list)
            dataset["/init_box_pose"] = BOX_POSE[0]
            dataset["/entropy"] = actions_entropy_norm
            dataset["/variance"] = actions_var_norm[:,None]
            dataset["/marginal_entropy"] = actions_marginal_var_norm
            path = f"data/act_replay/{task_name}"
            dataset_path = os.path.join(path, f"episode_{23+save_id}.hdf5")
            print(f"Saving to {dataset_path}")
            save_id += 1
            with h5py.File(dataset_path, "w") as root:  # 使用 "w" 模式，创建新文件
                for name, value in dataset.items():
                    try:
                        root.create_dataset(name, data=value)
                    except Exception as e:
                        print(f"Failed to save dataset '{name}': {e}")

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)
    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))
    
    return success_rate, avg_return

def eval_speed_bc(config, ckpt_name, H_dict,save_episode=True):
    set_seed(0)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
   
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
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha

        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from act.sim_env import make_sim_env
        
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config["num_queries"]
    query_frequency = 50//4
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    max_timesteps = max_timesteps//2
    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    max_entropy_list = []
    min_entropy_list = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if "sim_transfer_cube" in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset
        
        ts = env.reset()
 
        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_entropy = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 1]
            ).cuda()
            all_time_samples = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 10,state_dim]
            ).cuda()
 
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        traj_action_entropy = []
        flag = False
        waypoint_count = 0
        openloop_t = 0
        last_t = 0
        timestep_count = 0
        policy_slow = False
        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # get entropy
                        action_samples,_,action_trust = policy.get_entropy(qpos, curr_image)
                        action_samples = action_samples.squeeze().permute(1,0,2) # (chunk_len, num_samples, dim)
                        # all_actions = action_samples[:,-1].reshape(all_actions.shape)
                        # entropy = torch.mean(torch.var(action_samples,dim=1),dim=-1,keepdim=True)
                        # entropy = (entropy)*1e5 # 6e-7, 6e-5 for insertion
                        # entropy = torch.clip(entropy, 0, 1)
                        # all_actions = action_trust if weights >0 else all_actions
                        # get waypoints using entropy and actions
                            
                    if temporal_agg:
                        if not policy_slow:
                            # all_time_actions[[t], t : t + num_queries] = all_actions
                            all_speed_actions = all_actions[:,::3]
                            all_time_actions[[t], t:t + all_speed_actions.shape[1]] = all_speed_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
    
                            all_time_samples[[t], t:t+ all_speed_actions.shape[1]] = action_samples[::3]
                            actions_for_next_step = all_time_actions[:, t] # t+10
                            samples_populated = torch.all(
                                actions_for_next_step != 0, axis=1
                            )
                            samples_for_curr_step = all_time_samples[:, t]
                            samples_for_curr_step = samples_for_curr_step[samples_populated]
                            
                            entropy = torch.mean(torch.var(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            # entropy = (entropy_for_curr_step * exp_weights).sum(
                            #     dim=0, keepdim=True
                            # )
                            entropy = (math.log(torch.mean(entropy)+1e-8,1.5))
                            # entropy = torch.tensor(entropy).cuda()
                            # For transfer 2x
                            # entropy = torch.tensor((entropy+37)/32).cuda() # 20%： 0.8
                            # For insertion 2x
                            # entropy = torch.tensor((entropy+37)/34.5).cuda()  # 20%：0.7
                            # For transfer 3x
                            entropy = torch.tensor((entropy+37)/32).cuda() # 0.82
                            # For insertion 3x
                            # entropy = torch.tensor((entropy+37)/34.5).cuda()  # 20%：0.7
                            weights = 5*torch.clip(0.2-entropy.squeeze(),0,0.2)  # weights: 80%: 0 20% 0~1
                            # weights = 1 if weights >0 else 0
                            weights = weights.cpu().numpy()
                            k = 0.01 
                           
                        if (t>20 and entropy <0.82) or policy_slow:
                            
                            policy_slow = True
                            a=0 #_,actions_for_curr_step = KDE.kde_entropy(actions_for_curr_step.unsqueeze(0),k=13)
                            # slow policy
                            all_speed_actions = all_actions[:,::2]
                            all_time_actions[[t], t:t + all_speed_actions.shape[1]] = all_speed_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
    
                            all_time_samples[[t], t:t+ all_speed_actions.shape[1]] = action_samples[::2]
                            actions_for_next_step = all_time_actions[:, t] # t+10
                            samples_populated = torch.all(
                                actions_for_next_step != 0, axis=1
                            )
                            samples_for_curr_step = all_time_samples[:, t]
                            samples_for_curr_step = samples_for_curr_step[samples_populated]
                            
                            entropy = torch.mean(torch.var(samples_for_curr_step.flatten(0,1),dim=0),dim=-1)
                            exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            # entropy = (entropy_for_curr_step * exp_weights).sum(
                            #     dim=0, keepdim=True
                            # )
                            entropy = (math.log(torch.mean(entropy)+1e-8,1.5))
                            # entropy = torch.tensor(entropy).cuda()
                            # For transfer 2x
                            # entropy = torch.tensor((entropy+37)/32).cuda() # 20%： 0.8
                            # For insertion 2x
                            # entropy = torch.tensor((entropy+37)/34.5).cuda()  # 20%：0.7
                            # For transfer 3x
                            entropy = torch.tensor((entropy+37)/32).cuda() # 0.8
                            # For insertion 3x
                            # entropy = torch.tensor((entropy+37)/34.5).cuda()  # 20%：0.7
                            weights = 5*torch.clip(0.2-entropy.squeeze(),0,0.2)  # weights: 80%: 0 20% 0~1
                            # weights = 1 if weights >0 else 0
                            weights = weights.cpu().numpy()
                            k = 0.01 
                            # Change policy_slow flag if entropy is large
                            if entropy>0.82:
                                policy_slow = False
                        
                        # k = 0.01 
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                        traj_action_entropy.append(entropy.squeeze())
                            
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        entropy = torch.mean(torch.var(action_samples.squeeze(),dim=1),dim=-1)
                        # entropy = (entropy-H_dict["min"])/(H_dict["max"]-H_dict["min"])
                        entropy = [e for e in entropy]
                        traj_action_entropy.extend(entropy)
                        # raw_action = all_actions[:, openloop_t]
                        # openloop_t += 1

                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ### close-loop at gripper open/close destroy the performance
                ts = env.step(target_qpos)
                timestep_count += 1
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                # compensate for imperfect gripper
                  
                    
                if np.array(ts.reward) == env_max_reward:
                    timestep_count -= 1
                    # print(t)
                    break

            plt.close()
        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        print(f"Total time count:{timestep_count}")
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())
        print(traj_action_entropy.shape)
        print(f"max:{np.max(traj_action_entropy[:])} min:{np.min(traj_action_entropy[:])}")
        """if episode_highest_reward==env_max_reward:
            max_entropy_list.append(np.max(traj_action_entropy))
            min_entropy_list.append(np.min(traj_action_entropy))
        traj_action_entropy = (traj_action_entropy-np.min(traj_action_entropy))/(np.max(traj_action_entropy)-np.min(traj_action_entropy))
        """
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
        from act.act_utils import plot_3d_trajectory
        plot_3d_trajectory(ax1, left_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        plot_3d_trajectory(ax2, right_arm_xyz, traj_action_entropy,label="policy rollout", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        
        if save_episode:
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, f"video/video{rollout_id}.mp4"),
            )
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_view.png")
            )
        
        n_groups = qpos_numpy.shape[-1]
        tstep = np.linspace(0, 1, qpos_numpy.shape[0]-1) 
        fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)

        """for n, ax in enumerate(axes):
            ax.plot(tstep, np.array(qpos_list)[1:, n], label=f'real qpos {n}')
            ax.plot(tstep, np.array(target_qpos_list)[:-1, n], label=f'target qpos {n}')
            ax.set_title(f'qpos {n}')
            ax.legend()
        """
        plt.xlabel('timestep')
        plt.ylabel('qpos')
        plt.tight_layout()
        # fig.savefig(
        #         os.path.join(ckpt_dir, f"plot/rollout{rollout_id}_qpos.png")
        #     )
        plt.close()
        print(f"Save qpos curve to {ckpt_dir}/plot/rollout{rollout_id}_qpos.png")
        

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))
    print(f"max entropy:{np.min(np.array(max_entropy_list))} min entropy:{np.max(np.array(min_entropy_list))}")
    return success_rate, avg_return


def reset_env(env, initial_state=None, remove_obj=False):
    # load the initial state
    if initial_state is not None:
        env.reset_to(initial_state)

    # remove the object from the scene
    if remove_obj:
        remove_object(env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_demos", default=False, action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--visualize_entropy", action="store_true", help="visualize trajectory entropy", default=False
    )
    parser.add_argument(
        "--eval_speed", action="store_true", help="eval sped up policy", default=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--use_wandb", action="store_false")
    parser.add_argument("--plot", action="store_true")
    # for waypoints
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument(
        "--constant_waypoint",
        action="store",
        type=int,
        help="constant_waypoint",
        required=False,
    )

    main(vars(parser.parse_args()))