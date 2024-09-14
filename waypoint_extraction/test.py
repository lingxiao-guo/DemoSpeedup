import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

def compute_distance_to_curve(points, selected_indices):
    """计算所有点到选定点构成的曲线的最短距离"""
    selected_points = points[selected_indices]
    dists = []
    for p in points:
        min_dist = np.inf
        for i in range(len(selected_points) - 1):
            segment_start = selected_points[i]
            segment_end = selected_points[i + 1]
            # 计算点到线段的距离
            dist = np.linalg.norm(np.cross(segment_end - segment_start, p - segment_start) / np.linalg.norm(segment_end - segment_start))
            min_dist = min(min_dist, dist)
        dists.append(min_dist)
    return np.sum(dists)

def objective_function(selected_indices, points):
    """目标函数：选择的点数量"""
    return np.sum(selected_indices)

def constraint_function(selected_indices, points, threshold):
    """约束函数：计算距离总和，并确保其小于阈值"""
    selected_indices = np.where(selected_indices > 0.5)[0]  # 取整
    return threshold - compute_distance_to_curve(points, selected_indices)

def optimize_waypoint_selection(points, threshold):
    num_points = len(points)

    # 初始点选择：全0
    initial_guess = np.zeros(num_points)

    # 变量的边界
    bounds = [(0, 1) for _ in range(num_points)]

    # 约束：距离总和小于阈值
    constraints = {'type': 'ineq', 'fun': lambda x: constraint_function(x, points, threshold)}

    # 最小化目标函数
    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        args=(points,),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    
    selected_indices = np.where(result.x > 0.5)[0]
    return selected_indices
    
    # 变量的边界
    bounds = [(0, 1) for _ in range(num_points)]

    # 约束：距离总和小于阈值
    constraints = {'type': 'ineq', 'fun': lambda x: constraint_function(x, points, threshold)}

    # 最小化目标函数
    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        args=(points,),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    
    selected_indices = np.where(result.x > 0.5)[0]
    return selected_indices
    

# 假设原始点数据（400个点，每个点14维）
np.random.seed(42)
original_points = np.random.rand(400, 14)

# 设置阈值
threshold = 1

# 执行优化
selected_indices = optimize_waypoint_selection(original_points, threshold)
selected_points = original_points[selected_indices]

print(f"选择了 {len(selected_indices)} 个点")
print(f"这些点：\n{selected_points}")
