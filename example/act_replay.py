import torch
import numpy as np
import os
import h5py
os.environ['MUJOCO_GL'] = 'osmesa'
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import wandb
from act.constants import DT
from act.constants import PUPPET_GRIPPER_JOINT_OPEN
from act.act_utils import load_data, relabel_waypoints  # data functions
from act.act_utils import sample_box_pose, sample_insertion_pose  # robot functions
from act.act_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from act.visualize_episodes import save_videos
from act.sim_env import BOX_POSE,make_sim_env
from act.act_utils import put_text
import IPython

e = IPython.embed

############################## Usage: ##########################
"""
python example/act_replay.py --dataset data/act_replay/sim_transfer_cube_human --ckpt_dir data/outputs/act_ckpt/sim_transfer_cube_human_gaussian --task_name sim_transfer_cube_human --end_idx 9
"""

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    dataset_path = args["dataset"]
 
    # get task parameters
    # is_sim = task_name[:4] == 'sim_'
    is_sim = True  # hardcode to True to avoid finding constants from aloha
    if is_sim:
        from act.constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"

    config = {
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "onscreen_render": onscreen_render,
        "task_name": task_name,
        "seed": args["seed"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "start_idx": args["start_idx"],
        "end_idx": args["end_idx"],
        "use_waypoint": args["use_waypoint"],
        "use_entropy_waypoint": args["use_entropy_waypoint"],
        "use_constant_waypoint": args["use_constant_waypoint"]
    }
    is_eval = True
    if is_eval:
        ckpt_names = [f"policy_last.ckpt"]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = replay(config, ckpt_name, dataset_path,save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    

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

def is_in_bottom_20_percent(lst, elem):
    # 对列表进行排序
    sorted_lst = sorted(lst)
    
    # 计算前80%的位置索引
    bottom_20_index = int(len(sorted_lst)*0.4) # 1: 80% 0.8 0.5 0.3
    
    # 检查元素是否在后20%的范围内
    return elem <= sorted_lst[bottom_20_index]

def replay(config, ckpt_name, dataset, save_demos=False,save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    onscreen_render = config["onscreen_render"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    start_idx = config["start_idx"]
    end_idx = config["end_idx"]
    use_waypoint = config["use_waypoint"]
    use_entropy_waypoint = config["use_entropy_waypoint"]
    use_constant_waypoint = config["use_constant_waypoint"]
    onscreen_cam = "angle"
    variance_step = 1
   
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
        from act.act_utils import put_text, mark_3d_trajectory

        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
    

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = end_idx-start_idx+1
    episode_returns = []
    highest_rewards = []
    total_count = 0
    for rollout_id in range(start_idx, end_idx+1):
        rollout_id += 0
        dataset_path = os.path.join(dataset, f"episode_{rollout_id}.hdf5")
        with h5py.File(dataset_path, "r+") as root:
            all_qpos = root["/observations/qpos"][()]
            actions = root["/action"][()]
            entropy = root["/entropy"][()]
            waypoints = np.arange(0,len(actions))
            if use_waypoint:
                waypoints = root["/waypoints"][()]
                actions = np.array(actions)[waypoints]
                entropy = np.array(entropy)[waypoints]
                # actions = relabel_waypoints(actions, waypoints)
            elif use_entropy_waypoint:
                waypoints = root["/entropy_waypoints"][()]
                actions = np.array(actions)[waypoints]
                entropy = np.array(entropy)[waypoints]
            elif use_constant_waypoint:
                waypoints = list(np.arange(0,len(actions)))[::2]
                actions = np.array(actions)[waypoints]
                entropy = np.array(entropy)[waypoints]
            gripper_indices = gripper_change_detect(actions,all_qpos)
            images = root["/observations/images/top"][()]
            BOX_POSE[0] = root["/init_box_pose"][()]
        max_timesteps = int(np.array(actions).shape[0])
        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []

        ts = env.reset()
        count = 0
        last_flag = True
        last_count = 0
        mark_list = []
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
                curr_image = get_image(ts, camera_names)
                    
                ### store processed image for video 
                store_imgs = {}
                key_flag =  is_in_bottom_20_percent(entropy,entropy[t]) or (t in gripper_indices) 
                mark_list.append(key_flag)
                for key, img in obs["images"].items():
                    if t>0:
                        text = waypoints[t]-waypoints[t-1]
                        store_imgs[key] = put_text(img,np.concatenate(([text],[waypoints[t]])) )
                        if key_flag is True and mark_list[-2] is False:
                            store_imgs[key] = put_text(store_imgs[key],"******",is_waypoint=True,position="bottom")
                        elif key_flag:
                            store_imgs[key] = put_text(store_imgs[key],"*",position="bottom")
                        # if t<max_timesteps-1:
                        #      store_imgs[key] = put_text(store_imgs[key], all_qpos[t+1,[6,13]]-all_qpos[t,[6,13]],position="bottom")
                    else:
                        store_imgs[key] = img
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})

                target_qpos = actions[t]
                # TODO: Increase the end-effector force. Calculate delta, then increase the delta
                # target_qpos[[6,13]] = 0.5*(actions[t,[6,13]]+all_qpos[t,[6,13]])
                
                gripper_delta = target_qpos[[6,13]]-ts.observation["qpos"][[6,13]]
                # target_qpos[[6,13]] += gripper_delta*0.5
                non_gripper_idx = [0,1,2,3,4,5,7,8,9,11,12]
                ### step the environment
                ts = env.step(target_qpos)
                # ts = env.step(target_qpos)
                real_qpos = np.array(ts.observation["qpos"])
                count+=1
                if True: #is_in_bottom_20_percent(list(entropy),entropy[t]): 
                    closeloop_count = 0
                    # Consecutive two states don't have to be close-loop
                    # TODO: consider smooth motions
                    # Now can reach 75% 
                    # TODO: try to change the close-loop to interpolation
                    target_agent_qpos = target_qpos.copy()
                    target_agent_qpos[[6,13]] = all_qpos[t,[6,13]]
                    while  t>last_count+1 and closeloop_count<2 and np.linalg.norm(np.array(target_qpos-real_qpos)[non_gripper_idx],axis=-1)>0.05:
                        ts = env.step(target_qpos)
                        real_qpos = np.array(ts.observation["qpos"])
                        obs = ts.observation
                        qpos_numpy = np.array(obs["qpos"])
                        qpos = pre_process(qpos_numpy)
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                        curr_image = get_image(ts, camera_names)
                    
                        ### store processed image for video 
                        store_imgs = {}
                        for key, img in obs["images"].items():
                            if t>0:
                                text = 0
                                store_imgs[key] = put_text(img,np.concatenate(([text],[waypoints[t]])) )
                                if key_flag:
                                    store_imgs[key] = put_text(store_imgs[key],"*",position="bottom")
                            else:
                                store_imgs[key] = img
                        if "images" in obs:
                            image_list.append(store_imgs)
                        else:
                            image_list.append({"main": store_imgs})
                        count+=1
                        closeloop_count+=1
                        # print(f"target-real:{target_qpos[non_gripper_idx]-real_qpos[non_gripper_idx]}")
                        # print(count)
                        # print(f"target:{target_qpos[non_gripper_idx]}|real:{real_qpos[non_gripper_idx]}")
                        # print(f"target-real:{target_qpos[non_gripper_idx]-real_qpos[non_gripper_idx]}")
                    if closeloop_count>0:
                          last_count = t
                          last_flag = False
                    else:
                        last_flag = True
                else:
                    last_flag = True
                ### Compensate for the gripper delay
                ### Assume we have a perfect gripper controller
                

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

        mark_3d_trajectory(ax1, left_arm_xyz, mark_list,label="demos replay", legend=False)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Right", fontsize=20)
        ax2.set_xlim([min_x, max_x])
        ax2.set_ylim([min_y, max_y])
        ax2.set_zlim([min_z, max_z])

        mark_3d_trajectory(ax2, right_arm_xyz,mark_list,label="demos replay", legend=False)
        fig.suptitle(f"Task: {task_name}", fontsize=30) 

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=20)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}, Episode length:{count}"
        )
        total_count+=count        
        # Only save successful video/plot
        if save_episode : # and episode_highest_reward==env_max_reward: 
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(dataset, f"video/video{rollout_id}.mp4"),
            )
            fig.savefig(
                os.path.join(dataset, f"plot/plot_{rollout_id}.png")
            )
            ax1.view_init(elev=90, azim=45)
            ax2.view_init(elev=90, azim=45)
            fig.savefig(
                os.path.join(ckpt_dir, f"plot/plot_{rollout_id}_view.png")
            )
        plt.close(fig)
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
        fig.savefig(
                os.path.join(dataset, f"plot/rollout{rollout_id}_qpos.png")
            )
        print(f"Save qpos curve to {dataset}/plot/rollout{rollout_id}_qpos.png")    
        plt.close(fig)
        

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"
    
    print(summary_str)
    print(f"mean count: {total_count/(end_idx+1-start_idx)}")
    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))
    
    return success_rate, avg_return

def gripper_change_detect(actions, gt_states,err_threshold=0.01):
    gripper_change_indices = []
    for t in range(len(gt_states)-1):
        if any(np.abs(gt_states[t+1,[6,13]]-gt_states[t,[6,13]]) > err_threshold):
            gripper_change_indices.append(t)
    return gripper_change_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/act_replay/sim_transfer_cube_human",
        # default="data/act/sim_insertion_scripted",
        # default="data/act/sim_transfer_cube_human",
        # default="data/act/sim_insertion_human",
        # default="data/act/aloha_screw_driver",
        # default="data/act/aloha_coffee",
        # default="data/act/aloha_towel",
        # default="data/act/aloha_coffee_new",
        help="path to hdf5 dataset",
    )
    
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )

    parser.add_argument("--seed", action="store", type=int, help="seed", default=1000)
    
    parser.add_argument("--use_waypoint", action="store_true")
    parser.add_argument("--use_entropy_waypoint", action="store_true")
    parser.add_argument("--use_constant_waypoint", action="store_true")
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
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
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
    parser.add_argument("--use_wandb", action="store_false")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--constant_waypoint",
        action="store",
        type=int,
        help="constant_waypoint",
        required=False,
    )

    main(vars(parser.parse_args()))
