""" Automatic waypoint selection """
import numpy as np
from scipy.optimize import minimize, differential_evolution
import copy
from tqdm import tqdm
from waypoint_extraction.traj_reconstruction import (
    pos_only_geometric_waypoint_trajectory,
    pos_only_geometric_entropy_trajectory,
    pos_only_entropy_waypoint_trajectory,
    reconstruct_waypoint_trajectory,
    geometric_waypoint_trajectory,
    geometric_entropy_trajectory,
    calculate_weights_from_entropy,
    get_all_pos_only_geometric_distance,
    get_all_pos_only_geometric_distance_gpu,
    fast_pos_only_geometric_waypoint_trajectory
)


""" Iterative waypoint selection """
def greedy_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    geometry=True,
    pos_only=False,
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(len(actions) - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                waypoints.append(i)
                waypoints.append(i + 1)
        waypoints.sort()

    # reconstruct the trajectory, and record the reconstruction error for each state
    for i in range(len(actions)):
        if pos_only or geometry:
            func = (
                pos_only_geometric_waypoint_trajectory
                if pos_only
                else geometric_waypoint_trajectory
            )
            total_traj_err, reconstruction_error = func(
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                return_list=True,
            )
        else:
            _, reconstruction_error, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions,
                gt_states=gt_states,
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[0],
                remove_obj=remove_obj,
            )
        # break if the reconstruction error is below the threshold
        if total_traj_err < err_threshold:
            break
        # add the frame of the highest reconstruction error as a waypoint, excluding frames that are already waypoints
        max_error_frame = np.argmax(reconstruction_error)
        while max_error_frame in waypoints:
            reconstruction_error[max_error_frame] = 0
            max_error_frame = np.argmax(reconstruction_error)
        waypoints.append(max_error_frame)
        waypoints.sort()

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


def heuristic_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    geometry=True,
    pos_only=False,
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    for i in range(len(actions) - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            waypoints.append(i)
    waypoints.sort()

    # if 'robot0_vel_ang' or 'robot0_vel_lin' in gt_states is close to 0, make the frame a waypoint
    for i in range(len(gt_states)):
        if (
            np.linalg.norm(gt_states[i]["robot0_vel_ang"]) < err_threshold
            or np.linalg.norm(gt_states[i]["robot0_vel_lin"]) < err_threshold
        ):
            waypoints.append(i)

    waypoints.sort()

    print("=======================================================================")
    print(f"Selected {len(waypoints)} waypoints: {waypoints}")
    return waypoints


""" Backtrack waypoint selection """
def backtrack_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    # add heuristic waypoints
    num_frames = len(actions)

    # make the last frame a waypoint
    waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            waypoints.append(i)
    waypoints.sort()

    # backtracing to find the optimal waypoints
    start = 0
    while start < num_frames - 1:
        for end in range(num_frames - 1, 0, -1):
            rel_waypoints = [k - start for k in waypoints if k >= start and k < end] + [
                end - start
            ]
            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[start : end + 1],
                gt_states=gt_states[start + 1 : end + 2],
                waypoints=rel_waypoints,
                verbose=False,
                initial_state=initial_states[start],
                remove_obj=remove_obj,
            )
            if total_traj_err < err_threshold:
                waypoints.append(end)
                waypoints = list(set(waypoints))
                waypoints.sort()
                break
        start = end

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)))
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err,_ = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                return_list=True
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    # print(f"waypoint positions: {waypoints}")
    _,distance = pos_only_geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=True)
    return waypoints, distance

def optimize_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
    entropy=None,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
    
    from act.convert_ee import get_ee
    left_arm_ee = get_ee(actions[:, :6], actions[:, 6:7])
    right_arm_ee = get_ee(actions[:, 7:13], actions[:, 13:14])
    ee = np.concatenate([left_arm_ee[:,:6], right_arm_ee[:,:6]], axis=1)
    
    
    num_frames = len(actions)
    entropy_weights = calculate_weights_from_entropy(entropy)
    gripper_change_indices = gripper_change_detect(actions,gt_states)
    # entropy_weights[gripper_change_indices] = np.min(entropy_weights)*1.1
   
    # update err_threshold with entropy
    all_err_threshold = []
    for i in range(len(entropy_weights)):
        all_err_threshold.append(err_threshold*entropy_weights[i])
    all_err_threshold = np.array(all_err_threshold)
    
    # Get acceleration
    velocity = np.diff(gt_states,axis=0)
    acceleration_scale = np.max(np.linalg.norm(np.diff(velocity,axis=0),axis=-1))
    
    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        fast_pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )   
    all_distance = get_all_pos_only_geometric_distance_gpu(gt_states)
    print("All distances calculated.")
    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        min_smooth = float("inf")
        min_velocity = float("inf")
        best_waypoints = []
        
        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]
            # print(waypoints) # i=1, k=1, w=[499]+[1-1]=[0]  i=2,k=1,w=[2-1]=[1],k=2,w=
            total_traj_err,all_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                all_distance=all_distance[k : i + 1,k : i + 1,k : i + 1],
                return_list=True
            )
            # this cause local pareto!
            # TODO: add acceleration & rotation constraints!
            # TODO: change the dp to each segment optimization
            if (np.array(all_traj_err)<=all_err_threshold[k:i+1]).all():
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                total_waypoints_count = 1 + subproblem_waypoints_count
                # calculate the acceleration:
                candidate_waypoints = subproblem_waypoints + [i]
                smooth = get_smooth(candidate_waypoints,gt_states)
                if get_obj_func(total_waypoints_count,smooth,i) < get_obj_func(min_waypoints_required,min_smooth,i):
                        min_waypoints_required = total_waypoints_count
                        min_smooth = smooth
                        best_waypoints = candidate_waypoints 

        
        if min_waypoints_required < 1e6: 
            # prevent the initial inf to be given to memo[i]
            memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    distance,_ = func(actions, gt_states, waypoints, all_distance)
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {distance}"
    )
    print(f"waypoint positions: {waypoints}")
    return waypoints, distance

def get_acceleration(waypoints,gt_states):
    if len(gt_states)<3:
        return 0
    elif len(waypoints)<3:
        waypoints = [0] + [1] + waypoints
        waypoints = list(set(waypoints))
    else:
        if waypoints[0] != 0:
            waypoints = [0] + waypoints
    acceleration = np.mean(np.linalg.norm(np.diff(np.diff(gt_states[waypoints],axis=0),axis=0),axis=-1)) 
    return acceleration

def get_obj_func(waypoints_count, smooth,i, s_coeff=0.2): # 0.2
    # 0*inf is nan 
    obj_func =  waypoints_count/i + s_coeff*smooth 
    return obj_func
"""
def get_smooth(waypoints, gt_states,window_size=10):
    waypoints = gt_states[waypoints]
    waypoints = np.asarray(np.diff(waypoints,axis=0))  # 转换为 numpy 数组
    n = len(waypoints)
    rms_values = []  # 存储每个点的局部 RMS

    # 计算局部 RMS
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = waypoints[start:end]
        if len(window) > 0:
            rms = np.sqrt(np.mean(np.linalg.norm(window - waypoints[i],axis=-1)))
            rms_values.append(rms)

    # 如果没有有效的 RMS 值，则返回 0，否则返回 RMS 值的平均
    return np.mean(rms_values) if rms_values else 0
"""
def get_smooth(waypoints, gt_states):
    waypoints = [0] + waypoints
    waypoints = np.diff(waypoints)  
    smooth = np.var(waypoints)
    return smooth

def preprocess_entropy(data):
    # log -> zscore
    import math
    data = [math.log(d+1e-8) for d in data] # math.log(var+1e-8)
    data = np.array(data)
    data = (data - np.mean(data))/np.var(data)
    return data

""" Adaptive waypoint selection based on entropy """
# use geometric interpretation
def entropy_waypoint_selection(
    actions_entropy,
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    # preprocess entropy
    actions_entropy = preprocess_entropy(actions_entropy)
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(num_frames - 1):
            if actions[i, -1] != actions[i + 1, -1]:
                initial_waypoints.append(i)
                # initial_waypoints.append(i + 1)
        initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    obj_func = (
        pos_only_entropy_waypoint_trajectory
        if pos_only
        else geometric_entropy_trajectory
    )
    
    k = 255
    w_indices = None
    # w_indices = gripper_change_detect(actions, gt_states)
    # update k
    bounds = [(1/10, 10)] * k
     
    # initial_guess = np.linspace(0, gt_states.shape[0] - 2, k)
    initial_guess = (num_frames-1)/k * np.ones((k)) 
    

    """
    ##################### KNN for speed ######################
    
    from sklearn.cluster import KMeans
    waypoints = []
    num_clusters = 5  # 你可以根据需要调整这个数值
    
    # 初始化K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    # 计算速度
    velocity = actions[1:]-actions[:-1]
    velocity_norm = np.linalg.norm(velocity, axis=-1)
    # 计算加速度
    acceleration = velocity[1:] - velocity[:-1]
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    # 执行聚类
    kmeans.fit(velocity_norm[:,None])

    # 获取每个数据点的聚类索引
    cluster_indices = kmeans.labels_

    # 获取每个聚类的中心点
    cluster_centers = kmeans.cluster_centers_

    # 获取每个聚类的平均值
    # 由于K-means会计算每个簇的中心点，所以可以直接用这些中心点来表示平均值
    cluster_value_list = []
    cluster_indices_dict = {}
    cluster_means = cluster_centers.flatten()
    for cluster_id in range(num_clusters):
      if len(np.where(cluster_indices == cluster_id)[0]) >0:
            # print(f"Cluster {cluster_id}:")
            # print(f"  Mean value: {cluster_means[cluster_id]:.4f}")
            # print(f"  Indices: {np.where(cluster_indices == cluster_id)[0]}")
            # print(f"  Len: {len(np.where(cluster_indices == cluster_id)[0])}")
            cluster_value_list.append(cluster_means[cluster_id])
            cluster_indices_dict[f"{cluster_means[cluster_id]:.4f}"] = np.where(cluster_indices == cluster_id)[0]
    cluster_value_list.sort()
    l = len(cluster_indices_dict[f"{cluster_value_list[3]:.4f}"])+len(cluster_indices_dict[f"{cluster_value_list[4]:.4f}"])
    indices = []
    for i in range(num_clusters):
        if i==4:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(indices_temp[:])
        elif i==3:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(indices_temp[:])
        else:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            # waypoints.extend(indices_temp[::2])
            indices.extend(indices_temp)
    indices.sort()
    del kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    # 执行聚类
    kmeans.fit(velocity_norm[indices,None])

    # 获取每个数据点的聚类索引
    cluster_indices = kmeans.labels_

    # 获取每个聚类的中心点
    cluster_centers = kmeans.cluster_centers_

    # 获取每个聚类的平均值
    # 由于K-means会计算每个簇的中心点，所以可以直接用这些中心点来表示平均值
    cluster_value_list = []
    cluster_indices_dict = {}
    cluster_means = cluster_centers.flatten()
    for cluster_id in range(num_clusters):
      if len(np.where(cluster_indices == cluster_id)[0]) >0:
            print(f"Cluster {cluster_id}:")
            print(f"  Mean value: {cluster_means[cluster_id]:.4f}")
            # print(f"  Indices: {np.where(cluster_indices == cluster_id)[0]}")
            print(f"  Len: {len(np.where(cluster_indices == cluster_id)[0])}")
            cluster_value_list.append(cluster_means[cluster_id])
            cluster_indices_dict[f"{cluster_means[cluster_id]:.4f}"] = np.where(cluster_indices == cluster_id)[0]
    cluster_value_list.sort()
    l = 0
    m = []
    # print(f"Waypoints 01: {len(waypoints)}")
    # print(f"Indices: {len(indices)}")
    for i in range(num_clusters):
        if i==0:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(list(np.array(indices)[indices_temp[::3]]))
        elif i==1:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(list(np.array(indices)[indices_temp[::3]]))
        elif i==2:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(list(np.array(indices)[indices_temp[::2]]))
        elif i==3:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(list(np.array(indices)[indices_temp[::2]]))
        elif i==4:
            indices_temp = cluster_indices_dict[f"{cluster_value_list[i]:.4f}"]
            waypoints.extend(list(np.array(indices)[indices_temp[::2]]))

    
    ################# KNN for entropy #####################
    from sklearn.cluster import KMeans
    waypoints = []
    num_clusters = 5  # 你可以根据需要调整这个数值
    
    # 初始化K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    entropy_norm = len(actions_entropy)*actions_entropy-np.min(actions_entropy)/(np.max(actions_entropy)-np.min(actions_entropy))
    action_entropy_time = np.concatenate((entropy_norm[:,None],np.arange(0,len(actions_entropy))[:,None]),axis=-1)
    # 执行聚类
    kmeans.fit(action_entropy_time)

    # 获取每个数据点的聚类索引
    cluster_indices = kmeans.labels_

    # 获取每个聚类的中心点
    cluster_centers = kmeans.cluster_centers_

    # 获取每个聚类的平均值
    # 由于K-means会计算每个簇的中心点，所以可以直接用这些中心点来表示平均值
    cluster_time_list = []
    cluster_indices_dict = {}
    cluster_means = cluster_centers # .flatten()
    for cluster_id in range(num_clusters):
      if len(np.where(cluster_indices == cluster_id)[0]) >0:
            print(f"Cluster {cluster_id}:")
            print(f"  Mean value: {cluster_means[cluster_id]}")
            print(f"  Indices: {np.where(cluster_indices == cluster_id)[0]}")
            # print(f"  Len: {len(np.where(cluster_indices == cluster_id)[0])}")
            # cluster_value_list.append(cluster_means[cluster_id])
            cluster_time_list.append(cluster_means[cluster_id])
            cluster_indices_dict[f"{cluster_means[cluster_id]}"] = np.where(cluster_indices == cluster_id)[0]
    # cluster_value_list.sort()
    cluster_time_list.sort(key=lambda x: x[1])
    l = 0
    w_indices = []
    for i in range(num_clusters):
            # if i==0 or i==1:
            indices_temp = cluster_indices_dict[f"{cluster_time_list[i]}"]
            print(f"indices:{indices_temp}")
            if i ==0 and cluster_time_list[i][0]< cluster_time_list[i+1][0]:
                w_indices.append(indices_temp)
            elif i==num_clusters-1 and cluster_time_list[i][0]> cluster_time_list[i-1][0]:
                indices_temp[0:0] = [indices_temp[0]] * 9
                w_indices.append(indices_temp)
            elif (i>0 and i<num_clusters-1) and cluster_time_list[i][0]< cluster_time_list[i+1][0] and cluster_time_list[i][0]> cluster_time_list[i-1][0]:
                indices_temp[0:0] = [indices_temp[0]] * 9
                w_indices.append(indices_temp)
            
            l += len(indices_temp)
            print(f"new indices:{indices_temp}") 
            
            
        
    
    ################ SLSQP -> Actually return to Const Solution ####################
    def constraint(x, actions, sigma=2):
        # Constraints: choosing k points
        waypoints = (np.cumsum(x)-x[0])*399/np.sum(x)
        waypoints = np.round(waypoints).astype(int)
        waypoints = [w for w in waypoints]
        action_norm = np.linalg.norm(actions[1:]-actions[:-1], axis=-1)
        max_action_scale = np.max(action_norm)
        new_actions = actions[waypoints]
        new_action_norm = np.linalg.norm(new_actions[1:]-new_actions[:-1],axis=-1)
        penalty = np.clip(max_action_scale * sigma - new_action_norm,-1e6,0)
        return np.sum(penalty)
        # Next: add speed constraints and acceleration constraints! 
        # Acceleration could cause distinct force applied to objects!
    
    
    constraints = {'type': 'eq', 'fun': lambda x: constraint(x, actions)}
    result = minimize(
    obj_func,
    initial_guess,
    args=(actions, gt_states,  actions_entropy, w_indices),
    method='SLSQP',  # 'SLSQP'
    options={'disp': False},
    bounds=bounds,
    constraints=constraints,
    )
    """
    ##################### Differential Evolution #############################
    # """
    result = differential_evolution(
        obj_func,  # 自定义目标函数
        args=(actions, gt_states,  actions_entropy, w_indices),
        bounds=bounds,
        strategy='best1bin',  # 进化策略
        maxiter=10,         # 最大迭代次数
        popsize=15,           # 种群规模
        tol=1e-6,             # 收敛容忍度
        x0=initial_guess,
        disp=True             # 显示优化过程
    )
    # """
    _,waypoints = obj_func(result.x,actions, gt_states,  actions_entropy, w_indices,return_list=True )
    total_traj_err = result.fun
    
    
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints = list(waypoints)
    waypoints.sort()
    total_traj_err = obj_func(waypoints, actions, gt_states,  actions_entropy)
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")
    distance_func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )
    waypoints = [waypoint_i for waypoint_i in waypoints]
    waypoints.append(num_frames-1)
    _,distance = distance_func(actions, gt_states, waypoints, return_list=True)
    return waypoints, distance


def gripper_change_detect(actions, gt_states,err_threshold=0.01):
    gripper_change_indices = []
    for t in range(len(gt_states)-1):
        if any(np.abs(gt_states[t+1,[6,13]]-gt_states[t,[6,13]]) > err_threshold):
            gripper_change_indices.append(t)
    return gripper_change_indices


def dp_reconstruct_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            initial_waypoints.append(i)
    initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[k - 1 : i],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[k - 1],
                remove_obj=remove_obj,
            )

            print(f"i: {i}, k: {k}, total_traj_err: {total_traj_err}")

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

                    print(
                        f"min_waypoints_required: {min_waypoints_required}, best_waypoints: {best_waypoints}"
                    )

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(f"Minimum number of waypoints: {len(waypoints)}")
    print(f"waypoint positions: {waypoints}")

    return waypoints