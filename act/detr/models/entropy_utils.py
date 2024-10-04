import torch
import torch.nn.functional as F
from scipy.special import digamma
import numpy as np

def k_nn_distance(x, k):
    """
    计算每个样本到其 K 个最近邻的距离。
    
    Args:
    - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
    - k (int): 最近邻的数量
    
    Returns:
    - distances (torch.Tensor): 每个样本到其 K 个最近邻的距离，形状为 (batch_size, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    
    # 计算样本之间的距离
    x_flat = x.view(batch_size, num_samples, -1)
    distances = torch.cdist(x_flat, x_flat)  # (batch_size, num_samples, num_samples)
    
    # 计算每个样本的 K 最近邻的距离
    k_distances, _ = torch.topk(distances, k + 1, dim=-1, largest=False)
    k_distances = k_distances[:, :, 1:]  # 排除自身的距离
    
    return k_distances

def kozachenko_leonenko_entropy(x, k=5):
    """
    使用 Kozachenko-Leonenko 估计器估计连续随机变量的熵。
    
    Args:
    - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
    - k (int): 最近邻的数量
    
    Returns:
    - entropy (torch.Tensor): 估计得到的熵，形状为 (batch_size, 1)
    """
    batch_size, num_samples, dim = x.size()
    
    # 计算 K 最近邻距离
    k_distances = k_nn_distance(x, k)
    
    # 计算每个样本的平均距离
    avg_distances = k_distances.mean(dim=2)
    
    # 计算熵
    # 使用 Digamma 函数计算
    digamma_k = torch.tensor(digamma(k), dtype=torch.float32, device=x.device)
    digamma_n = torch.tensor(digamma(num_samples), dtype=torch.float32, device=x.device)
    
    entropy = digamma_n - digamma_k - dim * torch.log(avg_distances).mean(dim=1, keepdim=True)
    
    return entropy

def gaussian_kernel(x, bandwidth):
    """
    计算高斯核函数。

    Args:
    - x (torch.Tensor): 样本点，形状为 (batch_size, num_samples, dim)
    - bandwidth (float): 核函数的带宽（标准差）

    Returns:
    - kernel_values (torch.Tensor): 高斯核的计算值，形状为 (batch_size, num_samples, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    
    # 扩展维度以便计算距离矩阵
    x_i = x.unsqueeze(2)  # (batch_size, num_samples, 1, dim)
    x_j = x.unsqueeze(1)  # (batch_size, 1, num_samples, dim)
    
    # 计算距离矩阵
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (batch_size, num_samples, num_samples)
    
    # 计算高斯核
    kernel_values = torch.exp(-distances / (2 * bandwidth ** 2))
    
    return kernel_values


class KDE():
    def __init__(self, kde_flag=True, marginal_flag=True):
        self.flag = kde_flag
        self.marginal_flag = marginal_flag
    
    def kde_entropy(self,x,k=1):
        """
        使用核密度估计计算样本的熵，并对批次进行并行计算。

        Args:
        - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
        - bandwidth (float): 核函数的带宽（标准差）

        Returns:
        - entropy (torch.Tensor): 计算得到的熵，形状为 (batch_size, 1)
        """
        batch_size, num_samples, dim = x.size()
        # print(f"kde:{estimate_bandwidth(x[0])}")
        if self.flag:
            bandwidth = self.estimate_bandwidth(x[0])
            self.flag = False
        bandwidth = 0.001 # 0.002 for insertion, 0.001 for transfer
        # 计算高斯核
        kernel_values = gaussian_kernel(x, bandwidth)  # (batch_size, num_samples, num_samples)
    
        # 计算密度
        density = kernel_values.sum(dim=2) / num_samples  # (batch_size, num_samples)
        _, indices = torch.topk(density, k=k, dim=1)
        sorted_indices = indices.squeeze(0).sort(dim=0)[0]
        x_max_likelihood = x[0, sorted_indices, :]
        # 计算对数密度
        log_density = torch.log(density + 1e-8)  # 添加平滑项以避免 log(0)
        
        # 计算熵
        entropy = -log_density.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        return entropy, x_max_likelihood

    def kde_marginal_action_entropy(self,x):
        """
        使用核密度估计计算样本的熵，并对批次进行并行计算。

        Args:
        - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
        - bandwidth (float): 核函数的带宽（标准差）

        Returns:
        - entropy (torch.Tensor): 计算得到的熵，形状为 (batch_size, dim)
        """
        batch_size, num_samples, dim = x.size()
        # (batch_size*dim, num_samples, 1)
        x = x.permute(0,2,1).reshape(batch_size*dim, num_samples, 1)
        bandwidth = 0.0002 # 0.002 for insertion, 0.001 for transfer
        # 计算高斯核
        kernel_values = gaussian_kernel(x, bandwidth)  # (batch_size*dim, num_samples, num_samples)
    
        # 计算密度
        density = kernel_values.sum(dim=2) / num_samples  # (batch_size*dim, num_samples)
    
        # 计算对数密度
        log_density = torch.log(density + 1e-8)  # 添加平滑项以避免 log(0)
    
        # 计算熵
        entropy = -log_density.mean(dim=1, keepdim=True)  # (batch_size*dim, 1)
        entropy = entropy.reshape(batch_size, dim)
        return entropy

    def estimate_bandwidth(self,x, rule='scott'):
    
        num_samples, dim = x.size()
    
        std = x.std(dim=0).mean().item()  # 计算各维度的标准差的平均值
        if rule == 'silverman':
            bandwidth = 1.06 * std * num_samples**(-1/5)
        elif rule == 'scott':
            bandwidth = std * num_samples**(-1/(dim + 4))
        else:
            raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")
    
        return bandwidth
















