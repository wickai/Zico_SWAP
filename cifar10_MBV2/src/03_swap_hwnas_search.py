#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SWAP (Sample-Wise Activation Patterns) 的神经网络架构搜索实现
使用searchspace_AZNAS_ImageNet中的MobileNetV2搜索空间，支持多种数据集
集成HW-NAS-Bench用于评估架构耗电量，实现SWAP和耗电量的帕累托最优搜索
"""
import os
import sys
import time
import math
import random
import logging
import argparse
import numpy as np

# pandas导入与版本检查
import pandas as pd
try:
    # 检查pandas版本
    pd_version = pd.__version__
    print(f"使用pandas版本: {pd_version}")
    # 如果是较旧版本，可能需要特殊处理
    from pkg_resources import parse_version
    if parse_version(pd_version) < parse_version('1.0.0'):
        print("警告: 使用的pandas版本较旧，可能存在兼容性问题")
except Exception as e:
    print(f"pandas版本检查失败: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 导入新的搜索空间模块
from searchspace import MobileNetSearchSpace, count_parameters_in_MB, MBConv 

# 导入HW-NAS-Bench工具包
HWNASBenchAPI = None
try:
    from hw_nas_bench_api.hw_nas_bench_api import HWNASBenchAPI
except ImportError:
    try:
        from hw_nas_bench_api import HWNASBenchAPI
    except ImportError:
        logging.warning("无法导入HW-NAS-Bench-API，请确保已正确安装")

# 可视化工具
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("未安装matplotlib，将不能生成可视化图表")

# ===== 帕累托优化相关函数 =====

def is_dominated(p1, p2):
    """
    判断p1是否被p2支配
    如果p2在所有目标上都不比p1差，且至少在一个目标上比p1好，则p1被p2支配
    
    参数:
        p1, p2: 包含多个目标值的列表或元组，如[swap_score, power_consumption]
    
    返回:
        布尔值：True表示p1被p2支配
    """
    # 如果有任何一个值是None，则不考虑支配关系
    if p1 is None or p2 is None:
        return False
    
    # 对于最大化的目标（如SWAP），逻辑相反，转成最小化问题处理
    # SWAP最大化，功耗最小化
    better_some = False
    for i in range(len(p1)):
        # 如果是最大化目标，需要反转比较逻辑
        if i == 0:  # SWAP是最大化目标
            if p2[i] > p1[i]:
                better_some = True
            elif p2[i] < p1[i]:
                return False
        else:  # 功耗是最小化目标
            if p2[i] < p1[i]:
                better_some = True
            elif p2[i] > p1[i]:
                return False
    return better_some

def non_dominated_sort(population):
    """
    非支配排序，将population划分为不同的帕累托前沿
    
    参数:
        population: 包含个体的列表，每个个体应有fitness_values属性（列表形式）
    
    返回:
        fronts: 列表的列表，每个子列表包含一个帕累托前沿的索引
    """
    fronts = [[]]  # 第一个前沿
    dominated_count = [0] * len(population)  # 每个个体被支配的次数
    dominated_solutions = [[] for _ in range(len(population))]  # 每个个体支配的解
    
    # 对每个个体计算支配关系
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            
            # 确保fitness_values存在且是有效的列表
            if ('fitness_values' not in population[i] or 
                population[i]['fitness_values'] is None or
                'fitness_values' not in population[j] or
                population[j]['fitness_values'] is None):
                continue
            
            p1 = population[i]['fitness_values']
            p2 = population[j]['fitness_values']
            
            if is_dominated(p1, p2):  # p1被p2支配
                dominated_solutions[j].append(i)
                dominated_count[i] += 1
            elif is_dominated(p2, p1):  # p2被p1支配
                dominated_solutions[i].append(j)
                dominated_count[j] += 1
        
        if dominated_count[i] == 0:  # 不被任何人支配，属于第一前沿
            fronts[0].append(i)
    
    # 生成其他前沿
    i = 0
    while i < len(fronts) and len(fronts[i]) > 0:
        next_front = []
        for j in fronts[i]:
            for k in dominated_solutions[j]:
                dominated_count[k] -= 1
                if dominated_count[k] == 0:
                    next_front.append(k)
        i += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts

def crowding_distance(front, population):
    """
    计算拥挤度距离，用于在同一帕累托前沿中区分个体
    
    参数:
        front: 一个前沿的索引列表
        population: 包含个体的列表，每个个体应有fitness_values属性
    
    返回:
        distances: 拥挤度距离字典，键为个体索引
    """
    if len(front) <= 2:
        return {idx: float('inf') for idx in front}
    
    distances = {idx: 0 for idx in front}
    
    # 确保所有个体都有有效的fitness_values
    valid_front = []
    for idx in front:
        if ('fitness_values' in population[idx] and 
            population[idx]['fitness_values'] is not None and
            isinstance(population[idx]['fitness_values'], (list, tuple))):
            valid_front.append(idx)
    
    # 如果有效前沿少于3个个体，则赋予相同的无限距离
    if len(valid_front) <= 2:
        return {idx: float('inf') for idx in front}
    
    # 获取目标数量
    objectives = len(population[valid_front[0]]['fitness_values'])
    
    for obj in range(objectives):
        # 按目标值排序
        try:
            sorted_front = sorted(valid_front, key=lambda x: population[x]['fitness_values'][obj])
        except (TypeError, IndexError):
            # 如果排序出错，跳过此目标
            continue
        
        # 边界点设为无穷大
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        
        # 计算中间点的拥挤度距离
        obj_range = (
            population[sorted_front[-1]]['fitness_values'][obj] -
            population[sorted_front[0]]['fitness_values'][obj]
        )
        
        if obj_range == 0:
            continue
        
        for i in range(1, len(sorted_front) - 1):
            try:
                distances[sorted_front[i]] += (
                    population[sorted_front[i+1]]['fitness_values'][obj] -
                    population[sorted_front[i-1]]['fitness_values'][obj]
                ) / obj_range
            except (TypeError, IndexError, KeyError):
                # 如果计算出错，跳过
                continue
    
    return distances

# ============ 1) 模型评估函数 ============

@torch.no_grad()
def evaluate(model, loader, device):
    """计算Top-1和Top-5准确率"""
    # 确保模型处于评估模式
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    
    # 添加详细输出
    detailed_output = len(loader) <= 10  # 当测试集较小时提供详细输出
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # Top-1准确率
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(targets).sum().item()
        correct_top1 += batch_correct
        
        # Top-5准确率 (对于CIFAR-100更有意义)
        _, predicted_top5 = outputs.topk(5, 1, True, True)
        batch_correct_top5 = predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()
        correct_top5 += batch_correct_top5
        
        total += targets.size(0)
        
        # 输出详细信息
        if detailed_output:
            batch_acc = batch_correct / targets.size(0)
            print(f"Test Batch {batch_idx}/{len(loader)}: "
                  f"Acc@1={batch_acc*100:.2f}%, "
                  f"Batch Correct={batch_correct}/{targets.size(0)}")
    
    if total == 0:
        return 0.0, 0.0

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    
    # 添加更详细的输出
    print(f"评估结果: 总样本={total}, 正确预测Top1={correct_top1}, Top5={correct_top5}")
    print(f"Accuracy: Top1={top1_acc*100:.2f}%, Top5={top5_acc*100:.2f}%")
    
    return top1_acc, top5_acc

# ============ 2) 数据增强：Cutout实现 ============

class Cutout(object):
    """实现Cutout数据增强"""
    
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): 形状为(C, H, W)的张量
        Returns:
            带有n_holes个剪切块的图像
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), dtype=torch.float32)
        
        for n in range(self.n_holes):
            y = random.randint(0, h - self.length)
            x = random.randint(0, w - self.length)
            
            y1 = max(0, y)
            y2 = min(h, y + self.length)
            x1 = max(0, x)
            x2 = min(w, x + self.length)
            
            mask[y1:y2, x1:x2] = 0.
            
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


# ============ 3) mixup实现 ============

def mixup_data(x, y, alpha=1.0):
    """返回混合数据"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """返回mixup损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ 4) SWAP评估相关 ============

def cal_regular_factor(model, mu, sigma):
    """
    基于高斯形惩罚因子:
    factor = exp( -((param_kb - mu)^2) / (2 * sigma^2) )
    返回 float
    """
    param_mb = count_parameters_in_MB(model)
    param_kb = param_mb * 1e3
    
    # 建议引入 2.0 以更符合高斯分布
    #val = -((param_kb - mu) ** 2) / (2.0 * (sigma ** 2))
    val = -((param_kb - mu) ** 2) / sigma
    return math.exp(val)


class SampleWiseActivationPatterns:
    """
    SWAP 逻辑:
      - 收集ReLU输出(这里已兼容ReLU6)的 sign()
      - 转成 (F, N) 后做 unique(dim=0)
      - unique行数作为 SWAP 分数
    """
    def __init__(self, device):
        self.device = device
        self.activations = None

    @torch.no_grad()
    def collect_activations(self, feats):
        self.activations = feats.sign().to(self.device)

    @torch.no_grad()
    def calc_swap(self, reg_factor=1.0):
        if self.activations is None:
            return 0
        # 转置后 unique(dim=0)
        self.activations = self.activations.t()  # => (features, N)
        unique_patterns = torch.unique(self.activations, dim=0).size(0)
        return unique_patterns * reg_factor


class SWAP:
    """
    结合激活模式 + 模型参数量正则因子
    增加对功耗评估的支持
    """
    def __init__(self, device, regular=True, mu=None, sigma=None, hw_api=None, hw_target="edgegpu"):
        self.device = device
        self.regular = regular
        self.mu = mu
        self.sigma = sigma
        self.hw_api = hw_api  # HW-NAS-Bench API实例
        self.hw_target = hw_target  # 目标硬件平台
        self.fbnet_cache = {}  # 缓存查询结果

        self.inter_feats = []
        self.swap_evaluator = SampleWiseActivationPatterns(device)

    def evaluate(self, model, inputs):
        # 1) 计算正则因子
        if self.regular and (self.mu is not None) and (self.sigma is not None):
            reg_factor = cal_regular_factor(model, self.mu, self.sigma)
        else:
            reg_factor = 1.0

        # 2) 注册hook，抓取 ReLU / ReLU6 的输出
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                h = module.register_forward_hook(self._hook_fn)
                hooks.append(h)

        self.inter_feats = []

        # 3) 前向推理（只需要 1 次，无需训练）
        model.eval()
        with torch.no_grad():
            model(inputs.to(self.device))

        # 4) 计算SWAP
        if len(self.inter_feats) == 0:
            self._clear_hooks(hooks)
            return 0
        all_feats = torch.cat(self.inter_feats, dim=1)  # (N, sum_of_features)
        self.swap_evaluator.collect_activations(all_feats)
        swap_score = self.swap_evaluator.calc_swap(reg_factor)

        self._clear_hooks(hooks)
        self.inter_feats = []
        return swap_score
    
    def estimate_power(self, op_codes, width_codes):
        """
        使用HW-NAS-Bench估计架构的功耗
        
        参数:
            op_codes: 操作编码
            width_codes: 宽度编码
        
        返回:
            功耗估计值(mJ)，如果API不可用则基于简单估计
        """
        if self.hw_api is None:
            logging.debug("HW-NAS-Bench API不可用，使用简单估计功耗")
            return self._estimate_power_simple(op_codes, width_codes)
            
        try:
            # 生成缓存键
            cache_key = str(op_codes) + str(width_codes)
            if cache_key in self.fbnet_cache:
                return self.fbnet_cache[cache_key]
            
            # 将我们的架构编码转换为FBNet的编码格式
            fbnet_ops = self._convert_to_hwnas_encoding(op_codes)
            
            # 根据目标硬件选择对应的能耗指标键名
            energy_key = f"{self.hw_target}_energy"
            latency_key = f"{self.hw_target}_latency"
            
            # 方法1: 尝试直接查询
            try:
                hw_metrics = self.hw_api.query_by_index(fbnet_ops, dataname="cifar100")
                # 获取功耗信息（根据目标平台选择合适的指标）
                power = hw_metrics.get(energy_key, hw_metrics.get('energy', None))
                if power is not None:
                    self.fbnet_cache[cache_key] = power
                    return power
            except Exception as e:
                logging.debug(f"直接查询功耗失败: {e}")
            
            # 方法2: 寻找最相似架构
            try:
                # 只在第一次调用时获取所有架构
                if not hasattr(self, 'all_fbnet_archs'):
                    self.all_fbnet_archs = []
                    for i in range(1021):  # HW-NAS-Bench有1021个预计算架构
                        try:
                            config = self.hw_api.get_net_config(i, dataname="cifar100")
                            if config and 'op_idx_list' in config:
                                self.all_fbnet_archs.append((i, config['op_idx_list']))
                        except:
                            continue
                
                # 计算相似度并找出最相似的架构
                closest_idx = None
                min_distance = float('inf')
                
                for idx, arch_ops in self.all_fbnet_archs:
                    # 使用欧几里得距离
                    distance = sum((a-b)**2 for a, b in zip(fbnet_ops, arch_ops))
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                if closest_idx is not None:
                    # 查询最相似架构的性能
                    similar_metrics = self.hw_api.query_by_index(closest_idx, dataname="cifar100")
                    # 获取功耗信息
                    power = similar_metrics.get(energy_key, similar_metrics.get('energy', None))
                    if power is not None:
                        self.fbnet_cache[cache_key] = power
                        return power
            except Exception as e:
                logging.debug(f"查找相似架构失败: {e}")
            
            # 如果上述方法都失败，则使用简单估计
            return self._estimate_power_simple(op_codes, width_codes)
            
        except Exception as e:
            logging.warning(f"估计功耗时出错: {e}")
            return self._estimate_power_simple(op_codes, width_codes)
    
    def query_hwnas_accuracy(self, op_codes, width_codes, dataset="cifar100"):
        """
        从HW-NAS-Bench查询架构的accuracy
        
        参数:
            op_codes: 操作编码
            width_codes: 宽度编码
            dataset: 数据集名称
        
        返回:
            (accuracy, found): accuracy值和是否找到的布尔值
        """
        if self.hw_api is None:
            logging.debug("HW-NAS-Bench API不可用，无法查询accuracy")
            return None, False
            
        try:
            # 将我们的架构编码转换为FBNet的编码格式
            fbnet_ops = self._convert_to_hwnas_encoding(op_codes)
            
            # 方法1: 尝试直接查询
            try:
                hw_metrics = self.hw_api.query_by_index(fbnet_ops, dataname=dataset)
                # 获取accuracy信息
                accuracy = hw_metrics.get('valid_acc', hw_metrics.get('test_acc', hw_metrics.get('acc', None)))
                if accuracy is not None:
                    logging.info(f"直接从HW-NAS-Bench查询到accuracy: {accuracy:.4f}")
                    return accuracy, True
            except Exception as e:
                logging.debug(f"直接查询accuracy失败: {e}")
            
            # 方法2: 寻找最相似架构
            try:
                # 确保已经加载了所有架构
                if not hasattr(self, 'all_fbnet_archs'):
                    self.all_fbnet_archs = []
                    logging.info("正在加载HW-NAS-Bench中的所有架构...")
                    for i in range(1021):  # HW-NAS-Bench有1021个预计算架构
                        try:
                            config = self.hw_api.get_net_config(i, dataname=dataset)
                            if config and 'op_idx_list' in config:
                                self.all_fbnet_archs.append((i, config['op_idx_list']))
                        except:
                            continue
                    logging.info(f"加载了 {len(self.all_fbnet_archs)} 个HW-NAS-Bench架构")
                
                # 计算相似度并找出最相似的架构
                closest_idx = None
                min_distance = float('inf')
                
                for idx, arch_ops in self.all_fbnet_archs:
                    # 使用欧几里得距离
                    distance = sum((a-b)**2 for a, b in zip(fbnet_ops, arch_ops))
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                if closest_idx is not None:
                    # 查询最相似架构的性能
                    similar_metrics = self.hw_api.query_by_index(closest_idx, dataname=dataset)
                    accuracy = similar_metrics.get('valid_acc', similar_metrics.get('test_acc', similar_metrics.get('acc', None)))
                    if accuracy is not None:
                        logging.info(f"从最相似架构查询到accuracy: {accuracy:.4f} (距离: {min_distance:.2f})")
                        return accuracy, True
            except Exception as e:
                logging.debug(f"查找相似架构的accuracy失败: {e}")
            
            return None, False
            
        except Exception as e:
            logging.warning(f"查询accuracy时出错: {e}")
            return None, False
    
    def query_hwnas_metrics(self, op_codes, width_codes, dataset="cifar100"):
        """
        从HW-NAS-Bench查询架构的所有指标
        
        参数:
            op_codes: 操作编码
            width_codes: 宽度编码
            dataset: 数据集名称
        
        返回:
            (metrics_dict, found): 包含所有指标的字典和是否找到的布尔值
        """
        if self.hw_api is None:
            return {}, False
            
        try:
            # 将我们的架构编码转换为FBNet的编码格式
            fbnet_ops = self._convert_to_hwnas_encoding(op_codes)
            
            # 方法1: 尝试直接查询
            try:
                hw_metrics = self.hw_api.query_by_index(fbnet_ops, dataname=dataset)
                if hw_metrics:
                    logging.info("直接从HW-NAS-Bench查询到完整指标")
                    return hw_metrics, True
            except Exception as e:
                logging.debug(f"直接查询指标失败: {e}")
            
            # 方法2: 寻找最相似架构
            try:
                if not hasattr(self, 'all_fbnet_archs'):
                    self.all_fbnet_archs = []
                    logging.info("正在加载HW-NAS-Bench中的所有架构...")
                    for i in range(1021):
                        try:
                            config = self.hw_api.get_net_config(i, dataname=dataset)
                            if config and 'op_idx_list' in config:
                                self.all_fbnet_archs.append((i, config['op_idx_list']))
                        except:
                            continue
                    logging.info(f"加载了 {len(self.all_fbnet_archs)} 个HW-NAS-Bench架构")
                
                # 计算相似度并找出最相似的架构
                closest_idx = None
                min_distance = float('inf')
                
                for idx, arch_ops in self.all_fbnet_archs:
                    distance = sum((a-b)**2 for a, b in zip(fbnet_ops, arch_ops))
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                if closest_idx is not None:
                    similar_metrics = self.hw_api.query_by_index(closest_idx, dataname=dataset)
                    if similar_metrics:
                        logging.info(f"从最相似架构查询到指标 (距离: {min_distance:.2f})")
                        return similar_metrics, True
            except Exception as e:
                logging.debug(f"查找相似架构的指标失败: {e}")
            
            return {}, False
            
        except Exception as e:
            logging.warning(f"查询指标时出错: {e}")
            return {}, False
            
    def _convert_to_hwnas_encoding(self, op_codes):
        """
        将我们的架构编码转换为HW-NAS-Bench的FBNet编码格式的简单版本
        
        这个简化版本使用启发式规则将op_codes映射到FBNet操作
        不需要访问search_space对象
        
        FBNet操作索引对应关系：
        0: k3_e1    - 3x3 kernel, expand=1, group=1
        1: k3_e1_g2 - 3x3 kernel, expand=1, group=2
        2: k3_e3    - 3x3 kernel, expand=3, group=1
        3: k3_e6    - 3x3 kernel, expand=6, group=1
        4: k5_e1    - 5x5 kernel, expand=1, group=1
        5: k5_e1_g2 - 5x5 kernel, expand=1, group=2
        6: k5_e3    - 5x5 kernel, expand=3, group=1
        7: k5_e6    - 5x5 kernel, expand=6, group=1
        8: skip     - 跳过连接
        """
        # 简单启发式映射：按op_codes的值模9映射到[0-8]范围
        fbnet_ops = [(op % 9) for op in op_codes]
        
        # 确保长度为22，不足则用默认操作填充
        while len(fbnet_ops) < 22:
            fbnet_ops.append(0)  # 使用k3_e1填充
        
        # 如果超过22个，则截断
        if len(fbnet_ops) > 22:
            fbnet_ops = fbnet_ops[:22]
        
        return fbnet_ops
    
    def _estimate_power_simple(self, op_codes, width_codes):
        """
        简单的功耗估计方法，基于启发式规则
        
        参数:
            op_codes: 操作编码
            width_codes: 宽度编码
            
        返回:
            估计的功耗值(mJ)
        """
        # 计算权重因子：不同操作的复杂度不同
        op_weights = []
        for op in op_codes:
            # 根据操作复杂度设置权重（越复杂耗电越多）
            if op % 9 in [0, 1, 4, 5]:  # 低复杂度操作
                weight = 0.8
            elif op % 9 in [2, 6]:  # 中复杂度操作
                weight = 1.0
            elif op % 9 in [3, 7]:  # 高复杂度操作
                weight = 1.2
            else:  # skip连接
                weight = 0.5
            op_weights.append(weight)
        
        # 宽度因子：更宽的网络耗电更多
        width_factor = sum(width_codes) / len(width_codes) if width_codes else 1.0
        
        # 基本能耗模型
        base_energy = 100.0  # 基础能耗 mJ
        op_complexity_factor = sum(op_weights) / len(op_weights) if op_weights else 1.0
        
        # 最终估计的能耗
        estimated_energy = base_energy * op_complexity_factor * (1.0 + 0.5 * width_factor)
        
        return estimated_energy
            
    def _hook_fn(self, module, inp, out):
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)

    def _clear_hooks(self, hooks):
        for h in hooks:
            h.remove()
        hooks.clear() 

# ============ 5) 进化搜索 (SWAP作为评估) ============

class EvolutionarySearch:
    """
    多目标进化算法:
    - 使用SWAP和功耗作为优化目标
    - 实现非支配排序和拥挤度距离计算
    - 返回帕累托最优解集
    """
    def __init__(self, population_size, mutation_rate, n_generations,
                 swap_metric, search_space, device, num_inits=1,
                 tournament_size=3, diversity_weight=0.2):
        """
        :param num_inits: 对同一个结构多次随机初始化并计算SWAP平均，以减少初始化差异
        :param diversity_weight: 多样性权重，用于锦标赛选择
        """
        self.population_size = population_size
        self.initial_mutation_rate = mutation_rate  # 初始变异率
        self.current_mutation_rate = mutation_rate  # 当前变异率
        self.n_generations = n_generations
        self.swap_metric = swap_metric
        self.search_space = search_space
        self.device = device
        self.num_inits = num_inits
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight

    def update_mutation_rate(self, gen):
        """随着搜索进展减小变异率，AZNAS常用策略"""
        progress = gen / self.n_generations
        self.current_mutation_rate = self.initial_mutation_rate * (1 - 0.9 * progress)
        logging.info(f"  当前变异率: {self.current_mutation_rate:.4f}")
        
    def tournament_selection_multiobjective(self, population):
        """
        多目标锦标赛选择：
        1. 基于非支配排序级别选择
        2. 同级别中基于拥挤度距离选择
        """
        selected = []
        
        # 非支配排序，获取帕累托前沿
        fronts = non_dominated_sort(population)
        
        # 确保锦标赛大小不超过人口规模
        tournament_size = min(self.tournament_size, len(population))
        if tournament_size < 1:
            tournament_size = 1
        
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体
            candidates = random.sample(population, tournament_size)
            
            # 有时随机选择以增加多样性
            if random.random() < self.diversity_weight:
                best = random.choice(candidates)
            else:
                # 找出候选中的最优个体
                # 1. 找出候选中属于最高级别前沿的个体
                candidate_ranks = {}
                for i, front in enumerate(fronts):
                    for idx in front:
                        for j, cand in enumerate(candidates):
                            if cand is population[idx]:
                                candidate_ranks[j] = i
                
                best_rank = min(candidate_ranks.values())
                best_candidates = [candidates[j] for j, rank in candidate_ranks.items() if rank == best_rank]
                
                # 2. 如果有多个同级别的，则选择拥挤度距离最大的
                if len(best_candidates) > 1:
                    best_front_idx = -1
                    for i, front in enumerate(fronts):
                        if any(population[idx] in best_candidates for idx in front):
                            best_front_idx = i
                            break
                    
                    if best_front_idx != -1:
                        distances = crowding_distance(fronts[best_front_idx], population)
                        # 修复：安全处理None值
                        def safe_get_distance(x):
                            dist = distances.get(population.index(x), 0)
                            return dist if dist is not None else 0
                        best = max(best_candidates, key=safe_get_distance)
                    else:
                        best = random.choice(best_candidates)
                else:
                    best = best_candidates[0]
            
            selected.append(best)
        
        return selected

    def search(self, inputs):
        # 初始化多个子群体
        n_subpops = 3  # AZNAS通常使用3个子群体
        subpops = []
        
        # 创建多个子群体
        for _ in range(n_subpops):
            population = []
            for _ in range(self.population_size // n_subpops):
                op_codes = self.search_space.random_op_codes()
                width_codes = self.search_space.random_width_codes()
                population.append({
                    "op_codes": op_codes, 
                    "width_codes": width_codes, 
                    "fitness": None,  # 旧的单目标适应度，保留向后兼容
                    "fitness_values": [0, 0],  # 初始化为[0, 0]而不是None
                    "pareto_rank": None,  # 帕累托排序等级
                    "crowding_distance": None  # 拥挤度距离
                })
            subpops.append(population)
        
        for gen in range(self.n_generations):
            logging.info(f"=== Generation {gen+1} / {self.n_generations} ===")
            
            # 更新变异率
            self.update_mutation_rate(gen)
            
            # 处理每个子群体
            for subpop_idx, population in enumerate(subpops):
                logging.info(f"  子群体 {subpop_idx+1}/{n_subpops}:")
                
                # 评估适应度
                for i, indiv in enumerate(population):
                    if indiv["fitness_values"][0] == 0 and indiv["fitness_values"][1] == 0:
                        # 评估SWAP得分
                        swap_scores = []
                        for _ in range(self.num_inits):
                            model = self.search_space.build_model(indiv["op_codes"], indiv["width_codes"]).to(self.device)
                            # 简单初始化
                            for p in model.parameters():
                                if p.dim() > 1:
                                    nn.init.kaiming_normal_(p)
                            score = self.swap_metric.evaluate(model, inputs)
                            swap_scores.append(score)
                        swap_score = float(np.mean(swap_scores))
                        
                        # 评估功耗
                        power = self.swap_metric.estimate_power(indiv["op_codes"], indiv["width_codes"])
                        
                        # 设置多目标适应度和单目标适应度（向后兼容）
                        indiv["fitness_values"] = [swap_score, power]
                        indiv["fitness"] = swap_score  # 单目标模式下仍使用SWAP得分

                    logging.info(f"    [Ind-{i+1}] op_codes: {indiv['op_codes']}, width_codes: {indiv['width_codes']} | "
                               f"SWAP: {indiv['fitness_values'][0]:.3f}, Power: {indiv['fitness_values'][1]:.3f}")

                # 帕累托排序
                fronts = non_dominated_sort(population)
                
                # 更新每个个体的帕累托排名
                for rank, front in enumerate(fronts):
                    for idx in front:
                        population[idx]["pareto_rank"] = rank
                
                # 计算拥挤度距离
                for front in fronts:
                    distances = crowding_distance(front, population)
                    for idx, dist in distances.items():
                        population[idx]["crowding_distance"] = dist
                
                # 打印帕累托前沿个体
                logging.info(f"    第一帕累托前沿包含 {len(fronts[0])} 个解:")
                for idx in fronts[0][:min(5, len(fronts[0]))]:  # 只显示前5个，避免太多
                    ind = population[idx]
                    logging.info(f"      SWAP: {ind['fitness_values'][0]:.3f}, Power: {ind['fitness_values'][1]:.3f} | "
                               f"op_codes={ind['op_codes']}, width_codes={ind['width_codes']}")
                
                if gen < self.n_generations - 1:  # 最后一代不需要更新
                    # 使用多目标锦标赛选择
                    selected = self.tournament_selection_multiobjective(population)
                    
                    # 创建新一代
                    next_gen = []
                    
                    # 保留帕累托前沿的精英
                    elites_count = min(len(fronts[0]), max(1, self.population_size // 10))  # 取10%作为精英
                    for idx in fronts[0][:elites_count]:
                        next_gen.append(population[idx])
                    
                    # 生成新个体直到填满种群
                    while len(next_gen) < len(population):
                        # 随机选择父母
                        p1 = random.choice(selected)
                        p2 = random.choice(selected)
                        
                        # AZNAS风格的交叉：随机选择交叉策略
                        r = random.random()
                        if r < 0.4:  # 40%概率使用均匀交叉
                            child_op_codes = self.uniform_crossover(p1["op_codes"], p2["op_codes"])
                            child_width_codes = self.uniform_crossover(p1["width_codes"], p2["width_codes"])
                        elif r < 0.7:  # 30%概率使用单点交叉
                            child_op_codes = self.single_point_crossover(p1["op_codes"], p2["op_codes"])
                            child_width_codes = self.single_point_crossover(p1["width_codes"], p2["width_codes"])
                        else:  # 30%概率使用两点交叉
                            child_op_codes = self.two_point_crossover(p1["op_codes"], p2["op_codes"])
                            child_width_codes = self.two_point_crossover(p1["width_codes"], p2["width_codes"])
                        
                        # AZNAS变异：使用当前变异率
                        child_op_codes = self.mutate_op_codes(child_op_codes)
                        child_width_codes = self.mutate_width_codes(child_width_codes)
                        
                        next_gen.append({
                            "op_codes": child_op_codes,
                            "width_codes": child_width_codes,
                            "fitness": None,
                            "fitness_values": [0, 0],
                            "pareto_rank": None,
                            "crowding_distance": None
                        })
                    
                    # 更新种群
                    subpops[subpop_idx] = next_gen
            
            # 子群体之间的交流（每5代）
            if gen % 5 == 0 and gen > 0 and gen < self.n_generations - 1:
                logging.info("  执行子群体间交流...")
                for i in range(n_subpops):
                    next_i = (i + 1) % n_subpops
                    # 交换每个子群体的最佳个体（帕累托前沿）
                    best_idx = [idx for idx in range(len(subpops[i])) 
                               if subpops[i][idx].get("pareto_rank", float('inf')) == 0]
                    next_best_idx = [idx for idx in range(len(subpops[next_i])) 
                                    if subpops[next_i][idx].get("pareto_rank", float('inf')) == 0]
                    
                    if best_idx and next_best_idx:
                        # 随机选取帕累托前沿的个体交换
                        subpops[i].append(subpops[next_i][random.choice(next_best_idx)])
                        subpops[next_i].append(subpops[i][random.choice(best_idx)])
                        
                        # 移除多余个体以保持种群大小
                        # 更安全的方法：仅保留期望的种群大小
                        target_size = self.population_size // n_subpops
                        if len(subpops[i]) > target_size:
                            # 按照帕累托等级和拥挤度距离排序
                            def safe_sort_key(idx):
                                rank = subpops[i][idx].get("pareto_rank", float('inf'))
                                if rank is None:
                                    rank = float('inf')
                                distance = subpops[i][idx].get("crowding_distance", 0)
                                if distance is None:
                                    distance = 0
                                return (rank, -distance)
                            
                            sorted_indices = sorted(range(len(subpops[i])), key=safe_sort_key)
                            # 保留前target_size个个体
                            subpops[i] = [subpops[i][idx] for idx in sorted_indices[:target_size]]
                            
                        if len(subpops[next_i]) > target_size:
                            # 按照帕累托等级和拥挤度距离排序
                            def safe_sort_key_next(idx):
                                rank = subpops[next_i][idx].get("pareto_rank", float('inf'))
                                if rank is None:
                                    rank = float('inf')
                                distance = subpops[next_i][idx].get("crowding_distance", 0)
                                if distance is None:
                                    distance = 0
                                return (rank, -distance)
                            
                            sorted_indices = sorted(range(len(subpops[next_i])), key=safe_sort_key_next)
                            # 保留前target_size个个体
                            subpops[next_i] = [subpops[next_i][idx] for idx in sorted_indices[:target_size]]

        # 最后一代再做一遍评估，确保所有个体都有评估值
        all_individuals = []
        for population in subpops:
            for indiv in population:
                if indiv["fitness_values"][0] == 0 and indiv["fitness_values"][1] == 0:
                    # 评估SWAP得分
                    swap_scores = []
                    for _ in range(self.num_inits):
                        model = self.search_space.build_model(indiv["op_codes"], indiv["width_codes"]).to(self.device)
                        for p in model.parameters():
                            if p.dim() > 1:
                                nn.init.kaiming_normal_(p)
                        score = self.swap_metric.evaluate(model, inputs)
                        swap_scores.append(score)
                    swap_score = float(np.mean(swap_scores))
                    
                    # 评估功耗
                    power = self.swap_metric.estimate_power(indiv["op_codes"], indiv["width_codes"])
                    
                    indiv["fitness_values"] = [swap_score, power]
                    indiv["fitness"] = swap_score
                all_individuals.append(indiv)

        # 对所有个体进行帕累托排序
        fronts = non_dominated_sort(all_individuals)
        
        # 返回帕累托前沿的所有个体
        pareto_front = [all_individuals[idx] for idx in fronts[0]]
        
        # 记录所有帕累托最优解
        logging.info(f"搜索完成，找到 {len(pareto_front)} 个帕累托最优解:")
        for i, solution in enumerate(pareto_front):
            logging.info(f"  解 {i+1}: SWAP={solution['fitness_values'][0]:.3f}, Power={solution['fitness_values'][1]:.3f}")
        
        # 为了向后兼容，如果需要一个单一"最佳"解，选择SWAP得分最高的
        best = max(pareto_front, key=lambda x: x["fitness_values"][0])
        
        # 将整个帕累托前沿添加到结果中
        best["pareto_front"] = pareto_front
        
        return best

    def single_point_crossover(self, codes1, codes2):
        """单点交叉"""
        point = random.randint(1, len(codes1) - 1)
        return codes1[:point] + codes2[point:]
    
    def two_point_crossover(self, codes1, codes2):
        """两点交叉"""
        length = len(codes1)
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        return codes1[:point1] + codes2[point1:point2] + codes1[point2:]
    
    def uniform_crossover(self, codes1, codes2):
        """均匀交叉 - AZNAS常用"""
        child = []
        for i in range(len(codes1)):
            if random.random() < 0.5:
                child.append(codes1[i])
            else:
                child.append(codes2[i])
        return child
        
    def mutate_op_codes(self, op_codes):
        """变异操作码"""
        mutated = op_codes.copy()
        for i in range(len(mutated)):
            if random.random() < self.current_mutation_rate:  # 使用当前变异率
                mutated[i] = random.randrange(len(self.search_space.op_list))
        return mutated
        
    def mutate_width_codes(self, width_codes):
        """变异宽度码"""
        mutated = width_codes.copy()
        for i in range(len(mutated)):
            if random.random() < self.current_mutation_rate:  # 使用当前变异率
                mutated[i] = random.randrange(len(self.search_space.width_choices))
        return mutated 

# ============ 6) 最终训练 ============

def train_and_eval(model, train_loader, test_loader, device,
                   epochs=50, lr=0.01,
                   mixup_alpha=1.0,
                   label_smoothing=0.1):
    """
    对搜索到的最佳架构进行完整训练和评估
    
    使用多种训练技巧:
    - Mixup数据增强
    - 标签平滑正则化
    - Cutout(在DataLoader中设置)
    - 余弦退火学习率
    - 多GPU并行(如果可用)
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 训练设备
        epochs: 训练轮数
        lr: 初始学习率
        mixup_alpha: Mixup的alpha参数，如果为0则不使用Mixup
        label_smoothing: 标签平滑系数
    
    返回:
        最终测试集上的Top-1准确率
    """
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 多GPU数据并行
    if torch.cuda.device_count() > 1:
        print(f"=> 使用 {torch.cuda.device_count()} 个GPU进行训练...")
        model = nn.DataParallel(model)
    
    # 改进模型初始化 - 使用kaiming_normal替代原来的简单初始化
    print("执行模型重初始化...")
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:  # 检查bias是否存在
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:  # 检查weight是否存在
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # 检查bias是否存在
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:  # 检查bias是否存在
                nn.init.constant_(m.bias, 0)

    # 使用较小的学习率开始训练，防止梯度爆炸
    initial_lr = min(lr, 0.001)  # 从较小学习率开始
    print(f"初始学习率: {initial_lr}，目标学习率: {lr}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 使用带动量和权重衰减的优化器
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    
    # Warmup + 余弦退火学习率
    def warmup_cosine_schedule(epoch):
        warmup_epochs = min(10, epochs // 5)  # 预热阶段占总轮数的20%，但最多10轮
        if epoch < warmup_epochs:
            # 预热阶段线性增加学习率
            return (epoch + 1) / warmup_epochs * lr / initial_lr
        else:
            # 余弦退火阶段
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    best_acc = 0.0
    best_state = {'epoch': 0, 'state_dict': model.state_dict(), 'best_acc': best_acc}
    
    # 增加提前停止的计数器
    patience = 15  # 15轮无改善则提前停止
    patience_counter = 0
    
    # 记录训练和验证指标
    train_losses = []
    train_accs = []
    test_accs = []
    
    # 训练开始时间
    training_start_time = time.time()
    
    # 如果准确率长时间不提升，尝试调整学习率
    stagnant_counter = 0
    lr_adjustments = 0
    max_lr_adjustments = 3
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} 当前学习率: {current_lr:.6f}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup处理
            if mixup_alpha > 0.:
                mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                outputs = model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                # 不使用mixup
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            # 仅用于查看train top1（对mixup只是近似统计）
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            # 每100个批次打印一次训练情况
            if batch_idx % 100 == 0:
                batch_acc = preds.eq(labels).sum().item() / labels.size(0)
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {batch_acc*100:.2f}%")

        train_loss = total_loss / total
        train_acc_top1 = correct_top1 / total if total > 0 else 0.
        train_losses.append(train_loss)
        train_accs.append(train_acc_top1)

        # 在测试集上计算Top-1 / Top-5
        model.eval()  # 确保在评估前切换到评估模式
        test_top1, test_top5 = evaluate(model, test_loader, device)
        test_accs.append(test_top1)
        
        # 保存最佳模型
        if test_top1 > best_acc:
            best_acc = test_top1
            best_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }
            patience_counter = 0  # 重置提前停止计数器
            stagnant_counter = 0  # 重置停滞计数器
        else:
            patience_counter += 1
            stagnant_counter += 1
            
        # 如果准确率长时间停滞在低水平，尝试调整学习率
        if stagnant_counter >= 5 and test_top1 < 0.1 and lr_adjustments < max_lr_adjustments:
            # 减小学习率重试
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"警告: 准确率停滞不前，学习率调整为: {optimizer.param_groups[0]['lr']:.6f}")
            stagnant_counter = 0
            lr_adjustments += 1
            
            # 如果准确率极低，可能需要重新初始化网络
            if test_top1 < 0.02 and epoch > 10:
                print("重新初始化部分网络层...")
                # 仅重新初始化全连接层和最后几个卷积层
                for name, module in model.named_modules():
                    if 'classifier' in name or 'conv' in name and any(f'layer{i}' in name for i in [3, 4]):
                        if isinstance(module, nn.Conv2d):
                            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                            if module.bias is not None:  # 检查bias是否存在
                                nn.init.constant_(module.bias, 0)
                        elif isinstance(module, nn.Linear):
                            nn.init.normal_(module.weight, 0, 0.01)
                            if module.bias is not None:  # 检查bias是否存在
                                nn.init.constant_(module.bias, 0)
            
        # 计算每轮训练耗时
        epoch_time = time.time() - epoch_start_time
        
        # 计算预计剩余时间
        elapsed_time = time.time() - training_start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs
        
        # 计算分钟和秒
        mins, secs = divmod(estimated_remaining_time, 60)
        hours, mins = divmod(mins, 60)
        
        # 学习率
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Loss={train_loss:.3f}, "
                     f"Train@1={train_acc_top1*100:.2f}%, "
                     f"Test@1={test_top1*100:.2f}%, Test@5={test_top5*100:.2f}% | "
                     f"LR={current_lr:.6f}, Time={epoch_time:.1f}s, "
                     f"ETA={int(hours)}h {int(mins)}m {int(secs)}s")
        
        # 打印到控制台
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.3f}, Train Acc: {train_acc_top1*100:.2f}%, "
              f"Test Acc: {test_top1*100:.2f}%, Best: {best_acc*100:.2f}%")

        scheduler.step()
        
        # 提前停止
        if patience_counter >= patience and epochs > 50:
            logging.info(f"提前停止：连续 {patience} 轮没有改善")
            print(f"提前停止：连续 {patience} 轮没有改善")
            break
            
        # 如果准确率仍然很低，给出警告
        if epoch >= 10 and best_acc < 0.05:
            print("警告：训练10轮后准确率仍低于5%，模型可能存在架构问题或初始化问题")
            
        # 如果前5轮准确率没有提升，尝试更激进的学习率调整
        if epoch == 5 and best_acc < 0.02:
            print("前5轮准确率无明显提升，尝试更激进的学习率和优化器调整")
            # 尝试Adam优化器
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # 重新设置学习率调度器
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch)

    # 训练结束，计算总耗时
    total_time = time.time() - training_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成，总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 恢复最佳模型
    model.load_state_dict(best_state['state_dict'])
    final_top1, final_top5 = evaluate(model, test_loader, device)
    logging.info(f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    
    # 记录最佳精度轮次
    best_epoch = best_state['epoch']
    print(f"最佳测试准确率 (Epoch {best_epoch}): Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    
    # 如果有足够训练轮数，尝试绘制训练曲线
    if epochs > 5 and HAS_MATPLOTLIB:
        try:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accs)+1), [acc*100 for acc in train_accs], label='Train Acc')
            plt.plot(range(1, len(test_accs)+1), [acc*100 for acc in test_accs], label='Test Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'./logs/swap_search/training_curve.png')
            plt.close()
            print(f"训练曲线已保存到: ./logs/swap_search/training_curve.png")
        except Exception as e:
            print(f"绘制训练曲线失败: {e}")
    
    return final_top1


# ============ 7) 数据加载 ============

def get_dataset_dataloaders(dataset_name, root, batch_size, num_workers=2,
                           use_cutout=False, cutout_length=16):
    """
    创建不同数据集的训练和测试数据加载器
    
    参数:
        dataset_name: 数据集名称 ('cifar10', 'cifar100', 'imagenet')
        root: 数据存储路径
        batch_size: 批大小
        num_workers: 数据加载的工作进程数
        use_cutout: 是否使用Cutout数据增强
        cutout_length: Cutout方块的边长
    
    返回:
        (train_loader, test_loader, mean, std, num_classes, small_input)
    """
    if dataset_name == 'cifar10':
        # CIFAR-10配置
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        num_classes = 10
        dataset_train = datasets.CIFAR10
        dataset_test = datasets.CIFAR10
        img_size = 32
        small_input = True
        
    elif dataset_name == 'cifar100':
        # CIFAR-100配置
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
        dataset_train = datasets.CIFAR100
        dataset_test = datasets.CIFAR100
        img_size = 32
        small_input = True
        
    elif dataset_name == 'imagenet':
        # ImageNet配置
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        img_size = 224
        small_input = False
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 打印数据集信息
    print(f"数据集: {dataset_name}, 类别数: {num_classes}, 图像大小: {img_size}x{img_size}")
    print(f"均值: {mean}, 标准差: {std}")
    print(f"使用Cutout: {use_cutout}, Cutout长度: {cutout_length}" if use_cutout else "未使用Cutout")
    
    # 训练集数据增强
    if dataset_name in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            # 添加随机旋转
            transforms.RandomRotation(15),
            # 添加随机亮度和对比度调整
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        if use_cutout:
            train_transform.transforms.append(Cutout(n_holes=1, length=cutout_length))
        
        # 测试集数据处理
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # 创建数据集
        train_ds = dataset_train(root=root, train=True, download=True, transform=train_transform)
        test_ds = dataset_test(root=root, train=False, download=True, transform=test_transform)

    elif dataset_name == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # ImageNet需要指定文件夹路径
        train_ds = datasets.ImageFolder(
            os.path.join(root, 'train'),
            transform=train_transform
        )
        
        test_ds = datasets.ImageFolder(
            os.path.join(root, 'val'),
            transform=test_transform
        )
    
    # 验证数据集的类别分布
    if dataset_name in ['cifar10', 'cifar100']:
        print("验证数据集类别分布...")
        class_counts = [0] * num_classes
        for _, label in train_ds:
            class_counts[label] += 1
        
        print(f"训练集类别分布: 最小={min(class_counts)}个样本/类, 最大={max(class_counts)}个样本/类")
        # 检查是否均衡
        if min(class_counts) == max(class_counts):
            print("类别分布均衡")
        else:
            print("类别分布不均衡，但对于CIFAR数据集这是正常的")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 保留最后一个不完整批次
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"训练集: {len(train_ds)}个样本, {len(train_loader)}个批次")
    print(f"测试集: {len(test_ds)}个样本, {len(test_loader)}个批次")
    
    # 检查第一个批次以验证数据加载正确
    try:
        print("验证数据加载...")
        sample_batch, sample_labels = next(iter(train_loader))
        print(f"批次形状: {sample_batch.shape}, 标签形状: {sample_labels.shape}")
        print(f"标签范围: {sample_labels.min().item()} - {sample_labels.max().item()}")
        print(f"图像值范围: {sample_batch.min().item():.4f} - {sample_batch.max().item():.4f}")
    except Exception as e:
        print(f"数据加载验证失败: {e}")
    
    return train_loader, test_loader, mean, std, num_classes, small_input


# ============ 8) 计算 SWAP 正则所需的 mu,sigma 的函数 ============

def calculate_mu_sigma(search_space, n_samples=1000):
    """
    SWAP正则化所需的 mu & sigma
    这里我们取中位数做 mu，标准差做 sigma
    """
    arr = []
    for _ in range(n_samples):
        op_codes = search_space.random_op_codes()
        width_codes = search_space.random_width_codes()
        model = search_space.build_model(op_codes, width_codes)
        param_mb = count_parameters_in_MB(model)
        param_kb = param_mb * 1e3
        arr.append(param_kb)

    # 用 pandas 的 describe()
    series_ = pd.Series(arr)
    desc = series_.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    # 中位数
    mu = desc["50%"]
    # 标准差
    sigma = desc["std"]
    return mu, sigma


# ============ 9) 日志设置 ============

def setup_logger(log_path):
    """
    设置日志记录器，将日志同时输出到文件和控制台
    
    参数:
        log_path: 日志文件保存路径
    """
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_path, "swap_search.log"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    ) 

# ============ 10) 参数解析 ============

def parse_args():
    parser = argparse.ArgumentParser("MobileNetV2 Architecture Search with SWAP")
    parser.add_argument("--dataset", type=str, default="cifar100", 
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help="选择数据集: cifar10, cifar100, imagenet")
    parser.add_argument("--log_path", default="./logs/swap_search", type=str, help="where to save logs")
    parser.add_argument("--data_path", default="./data", type=str, help="数据集路径")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")

    # HW-NAS-Bench相关参数
    parser.add_argument("--hwnas_path", default="./hwnas_data", type=str, 
                       help="HW-NAS-Bench数据路径，需包含HW-NAS-Bench-v1_0.pickle文件")
    parser.add_argument("--hwnas_target", default="edgegpu", type=str, 
                       choices=['edgegpu', 'raspi4', 'eyeriss', 'pixel3'],
                       help="目标硬件平台，用于从HW-NAS-Bench中选择相应的功耗数据")
    parser.add_argument("--save_pareto", action="store_true", 
                       help="是否保存所有帕累托最优解")
    parser.add_argument("--power_weight", type=float, default=0.5,
                       help="功耗在多目标优化中的权重，值越大表示越注重功耗优化")
    parser.add_argument("--visualize", action="store_true",
                       help="是否绘制帕累托前沿可视化图")

    # Evolutionary search params
    parser.add_argument("--population_size", default=100, type=int, help="population size for evolutionary search")
    parser.add_argument("--mutation_rate", default=0.1, type=float, help="mutation prob")
    parser.add_argument("--n_generations", default=80, type=int, help="generations for ES")
    parser.add_argument("--search_batch", default=64, type=int,
                        help="batchsize for SWAP evaluation")

    # SWAP + param regular
    parser.add_argument("--use_param_regular", action="store_true", help="use SWAP with param regular factor")
    parser.add_argument("--n_samples_mu_sigma", default=1000, type=int, help="for random sampling to calc mu & sigma")
    parser.add_argument("--num_inits", default=3, type=int,
                        help="number of times to random-init the model for averaging SWAP score")

    # final training
    parser.add_argument("--train_batch", default=96, type=int, help="batchsize for final training")
    parser.add_argument("--train_epochs", default=200, type=int, help="epochs for final training")
    parser.add_argument("--lr", default=0.025, type=float, help="initial lr for final training")
    parser.add_argument("--skip_training", action="store_true", help="跳过最终训练步骤，仅进行架构搜索")

    # cutout
    parser.add_argument("--use_cutout", action="store_true", help="enable cutout in data augmentation")
    parser.add_argument("--cutout_length", default=16, type=int, help="cutout length")

    # mixup & label smoothing
    parser.add_argument("--mixup_alpha", default=0.8, type=float, help="mixup alpha, if 0 then no mixup")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing factor")

    # 添加锦标赛大小参数
    parser.add_argument("--tournament_size", default=3, type=int, 
                       help="tournament size for selection in evolutionary search")

    # HW-NAS-Bench查询参数
    parser.add_argument("--query_pareto_hwnas", action="store_true",
                       help="是否为帕累托前沿的所有解查询HW-NAS-Bench accuracy（可能较耗时）")

    args = parser.parse_args()
    return args 

# ============ 11) 可视化函数 ============

def plot_pareto_front(pareto_front, save_path='./logs/pareto_front.png'):
    """
    绘制帕累托前沿的可视化图
    
    参数:
        pareto_front: 帕累托最优解列表
        save_path: 图像保存路径
    """
    if not HAS_MATPLOTLIB:
        logging.warning("未安装matplotlib，无法生成可视化图表")
        return
    
    # 提取SWAP分数和功耗值
    swap_scores = []
    power_values = []
    op_complexity = []  # 用操作复杂度代替参数量
    
    for sol in pareto_front:
        swap_scores.append(sol['fitness_values'][0])
        power_values.append(sol['fitness_values'][1])
        
        # 计算操作复杂度（基于操作码数量）
        op_count = len(sol['op_codes'])
        op_complexity.append(op_count)
    
    # 创建一个新的图形
    plt.figure(figsize=(12, 10))
    
    # 使用英文标签代替中文，避免字体问题
    
    # 1. SWAP vs 功耗的散点图
    plt.subplot(2, 2, 1)
    sc = plt.scatter(swap_scores, power_values, c=range(len(swap_scores)), 
               cmap='viridis', alpha=0.8, s=100)
    
    plt.colorbar(sc, label='Solution Index')
    plt.xlabel('SWAP Score (Higher is Better)')
    plt.ylabel('Power (mJ) (Lower is Better)')
    plt.title('SWAP vs Power Pareto Front')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 标记最高SWAP和最低功耗的点
    max_swap_idx = np.argmax(swap_scores)
    min_power_idx = np.argmin(power_values)
    
    plt.annotate(f"Highest SWAP", xy=(swap_scores[max_swap_idx], power_values[max_swap_idx]),
                xytext=(swap_scores[max_swap_idx], power_values[max_swap_idx]*1.1),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.annotate(f"Lowest Power", xy=(swap_scores[min_power_idx], power_values[min_power_idx]),
                xytext=(swap_scores[min_power_idx]*0.9, power_values[min_power_idx]),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    # 2. 操作复杂度 vs 功耗的散点图
    plt.subplot(2, 2, 2)
    sc2 = plt.scatter(op_complexity, power_values, c=range(len(op_complexity)), 
                cmap='viridis', alpha=0.8, s=100)
    
    plt.colorbar(sc2, label='Solution Index')
    plt.xlabel('Operation Complexity (Op Count)')
    plt.ylabel('Power (mJ)')
    plt.title('Operation Complexity vs Power')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 操作复杂度 vs SWAP的散点图
    plt.subplot(2, 2, 3)
    sc3 = plt.scatter(op_complexity, swap_scores, c=range(len(op_complexity)), 
                cmap='viridis', alpha=0.8, s=100)
    
    plt.colorbar(sc3, label='Solution Index')
    plt.xlabel('Operation Complexity (Op Count)')
    plt.ylabel('SWAP Score')
    plt.title('Operation Complexity vs SWAP')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 3D图：SWAP vs 功耗 vs 操作复杂度
    ax = plt.subplot(2, 2, 4, projection='3d')
    sc4 = ax.scatter(swap_scores, power_values, op_complexity, 
               c=range(len(swap_scores)), cmap='viridis', s=100)
    
    plt.colorbar(sc4, label='Solution Index')
    ax.set_xlabel('SWAP Score')
    ax.set_ylabel('Power (mJ)')
    ax.set_zlabel('Operation Complexity (Op Count)')
    ax.set_title('SWAP vs Power vs Operation Complexity')
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logging.info(f"帕累托前沿可视化图已保存至 {save_path}")

# ============ 12) 主函数 ============

def main():
    args = parse_args()
    
    # 根据数据集名称创建日志目录
    log_path = os.path.join(args.log_path, f"swap_power_{args.dataset}")
    setup_logger(log_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    logging.info(f"使用数据集: {args.dataset}")
    
    # 添加调试信息
    print(f"开始执行多目标架构搜索...")

    # 1) 加载HW-NAS-Bench API
    hw_api = None
    try:
        logging.info(f"加载HW-NAS-Bench API，目标硬件: {args.hwnas_target}")
        # 检查HWNASBenchAPI是否成功导入
        if HWNASBenchAPI is None:
            logging.warning("HWNASBenchAPI未成功导入，无法使用HW-NAS-Bench")
        else:
            # 检查HW-NAS-Bench数据文件是否存在
            hw_file = os.path.join(args.hwnas_path, "HW-NAS-Bench-v1_0.pickle")
            if not os.path.exists(hw_file):
                logging.warning(f"HW-NAS-Bench数据文件不存在: {hw_file}")
                logging.warning("请下载HW-NAS-Bench-v1_0.pickle数据文件并放在正确位置")
            else:
                hw_api = HWNASBenchAPI(hw_file, search_space="fbnet")  # FBNet覆盖MobileNet算子
                logging.info(f"HW-NAS-Bench加载成功")
    except Exception as e:
        logging.warning(f"加载HW-NAS-Bench失败: {e}")
        logging.warning("将使用简单估计功耗")
    
    print("准备加载数据集...")

    # 2) 加载数据集并获取配置信息
    train_loader, test_loader, mean, std, num_classes, small_input = get_dataset_dataloaders(
        dataset_name=args.dataset,
        root=args.data_path,
        batch_size=args.train_batch,
        num_workers=4,
        use_cutout=args.use_cutout,
        cutout_length=args.cutout_length
    )
    
    print("数据集加载完成，创建搜索空间...")
    
    # 3) 创建与数据集匹配的搜索空间
    search_space = MobileNetSearchSpace(num_classes=num_classes, small_input=small_input)
    logging.info(f"搜索空间创建完成，使用MobileNetV2架构，针对{args.dataset}数据集")
    
    # 输出搜索空间信息
    logging.info(f"搜索空间: 共 {search_space.total_blocks} 个块, 宽度选项: {search_space.width_choices}")
    logging.info(f"操作类型: {search_space.op_list}")
    logging.info(f"网络结构: {search_space.stage_setting}")
    logging.info(f"类别数: {num_classes}, 小输入: {small_input}")

    # 4) 如果要用SWAP的 param regular，就先随机采样，计算 mu & sigma
    mu, sigma = None, None
    if args.use_param_regular:
        logging.info(f"采样 {args.n_samples_mu_sigma} 个架构来计算参数正则化的 mu & sigma ...")
        mu, sigma = calculate_mu_sigma(search_space, n_samples=args.n_samples_mu_sigma)
        logging.info(f"[SWAP param regular] mu={mu:.1f}, sigma={sigma:.1f}")

    print("准备构造评估批次...")
    
    # 5) 构造一个小批量数据 (search_batch) 用于SWAP评估
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            dataset_class = datasets.CIFAR10
        else:
            dataset_class = datasets.CIFAR100
        
        eval_ds = dataset_class(root=args.data_path, train=True, download=True, transform=transform_eval)
        search_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=args.search_batch, shuffle=True, num_workers=4
        )
    else:  # ImageNet
        eval_ds = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=transform_eval
        )
        search_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=args.search_batch, shuffle=True, num_workers=4
        )
    
    mini_inputs, _ = next(iter(search_loader))
    mini_inputs = mini_inputs.to(device)
    print(f"评估批次准备完成，批次大小: {mini_inputs.shape}")

    # 6) 构建 SWAP 指标（带有功耗评估）
    swap_metric = SWAP(device=device, regular=args.use_param_regular, mu=mu, sigma=sigma, hw_api=hw_api, hw_target=args.hwnas_target)
    print("SWAP评估器创建完成")

    # 7) 进化搜索
    print(f"开始进化搜索，种群大小: {args.population_size}，迭代次数: {args.n_generations}")
    es = EvolutionarySearch(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        n_generations=args.n_generations,
        swap_metric=swap_metric,
        search_space=search_space,
        device=device,
        num_inits=args.num_inits,
        tournament_size=args.tournament_size,
        diversity_weight=0.2  # AZNAS风格的多样性权重
    )
    
    start = time.time()
    try:
        print("开始搜索过程...")
        best_result = es.search(mini_inputs)
        print("搜索完成！")
    except Exception as e:
        logging.error(f"搜索过程出错: {e}")
        import traceback
        traceback.print_exc()
        return
    end = time.time()

    print(f"✅ 搜索成功完成，耗时 {end - start:.2f}秒")
    print("正在处理搜索结果...")
    
    logging.info(f"搜索完成，耗时 {end - start:.2f}秒.")
    
    # 获取帕累托前沿
    print("获取帕累托前沿...")
    pareto_front = best_result.get("pareto_front", [best_result])
    print(f"找到 {len(pareto_front)} 个帕累托最优解")
    
    # 输出帕累托前沿信息
    logging.info(f"找到 {len(pareto_front)} 个帕累托最优解:")
    
    # 创建一个数据表格以便查看结果
    pareto_data = []
    print("\n=== 帕累托最优解 ===")
    
    if not args.query_pareto_hwnas and hw_api is not None:
        print("💡 提示: 使用 --query_pareto_hwnas 参数可查询所有解的HW-NAS-Bench accuracy")
    elif args.query_pareto_hwnas:
        print("🔍 正在查询所有解的HW-NAS-Bench accuracy...")
        
    # 根据数据集选择合适的查询名称
    hwnas_dataset = args.dataset
    if args.dataset == "cifar10":
        hwnas_dataset = "cifar10"
    elif args.dataset == "cifar100":
        hwnas_dataset = "cifar100"
    elif args.dataset == "imagenet":
        hwnas_dataset = "imagenet"
    
    for i, solution in enumerate(pareto_front):
        swap_score = solution["fitness_values"][0]
        power = solution["fitness_values"][1]
        
        # 查询HW-NAS-Bench中的accuracy
        hwnas_acc = None
        if args.query_pareto_hwnas:  # 只有用户指定才查询
            try:
                if hw_api is not None:  # 确保API可用
                    hwnas_acc, found = swap_metric.query_hwnas_accuracy(
                        solution["op_codes"], 
                        solution["width_codes"], 
                        dataset=hwnas_dataset
                    )
                    if not found:
                        hwnas_acc = None
            except Exception as e:
                logging.debug(f"查询解{i+1}的HW-NAS-Bench accuracy失败: {e}")
                hwnas_acc = None
        
        pareto_data.append({
            "ID": i+1,
            "SWAP": swap_score,
            "Power": power,
            "HW-NAS-Bench_Accuracy": hwnas_acc,
            "op_codes": solution["op_codes"],
            "width_codes": solution["width_codes"]
        })
        
        print(f"解 {i+1}: SWAP={swap_score:.3f}, Power={power:.3f}", end="")
        if hwnas_acc is not None:
            print(f", HW-NAS-Bench Acc={hwnas_acc*100:.2f}%")
        else:
            print(f", HW-NAS-Bench Acc=N/A")
        print(f"  op_codes: {solution['op_codes']}")
        print(f"  width_codes: {solution['width_codes']}")
        
        logging.info(f"  解 {i+1}: SWAP={swap_score:.3f}, Power={power:.3f}" + 
                    (f", HW-NAS-Bench Acc={hwnas_acc*100:.2f}%" if hwnas_acc is not None else ", HW-NAS-Bench Acc=N/A"))
    
    # 保存帕累托前沿
    if args.save_pareto:
        print("保存帕累托前沿到CSV文件...")
        pareto_df = pd.DataFrame(pareto_data)
        pareto_file = os.path.join(log_path, f"pareto_front_{args.dataset}.csv")
        
        # 尝试多种方式保存数据
        try:
            # 方法1: 使用pandas to_csv
            pareto_df.to_csv(pareto_file, index=False)
        except ImportError as e:
            print(f"pandas to_csv失败: {e}")
            try:
                # 方法2: 使用numpy保存
                np.savetxt(
                    pareto_file.replace('.csv', '.txt'), 
                    np.array([[p['ID'], p['SWAP'], p['Power']] for p in pareto_data]),
                    fmt='%d,%.3f,%.3f', 
                    header='ID,SWAP,Power', 
                    comments=''
                )
                print(f"已使用numpy保存到: {pareto_file.replace('.csv', '.txt')}")
                
                # 同时保存完整数据为JSON
                import json
                json_file = pareto_file.replace('.csv', '.json')
                # 为JSON序列化处理numpy数组
                json_data = []
                for p in pareto_data:
                    p_copy = p.copy()
                    p_copy['SWAP'] = float(p_copy['SWAP'])
                    p_copy['Power'] = float(p_copy['Power'])
                    if p_copy['HW-NAS-Bench_Accuracy'] is not None:
                        p_copy['HW-NAS-Bench_Accuracy'] = float(p_copy['HW-NAS-Bench_Accuracy'])
                    json_data.append(p_copy)
                    
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print(f"完整数据已保存为JSON: {json_file}")
                
                pareto_file = pareto_file.replace('.csv', '.txt')
            except Exception as e2:
                print(f"使用numpy保存也失败: {e2}")
                print("尝试最基本的文本保存方式...")
                
                # 方法3: 使用基本的文本文件保存
                with open(pareto_file.replace('.csv', '.txt'), 'w') as f:
                    f.write("ID,SWAP,Power\n")
                    for p in pareto_data:
                        f.write(f"{p['ID']},{p['SWAP']:.3f},{p['Power']:.3f}\n")
                print(f"已使用基本文本方式保存到: {pareto_file.replace('.csv', '.txt')}")
                pareto_file = pareto_file.replace('.csv', '.txt')
        
        print(f"帕累托前沿已保存至: {pareto_file}")
        logging.info(f"帕累托前沿已保存至 {pareto_file}")
        
        # 打印统计信息
        hwnas_available = pareto_df['HW-NAS-Bench_Accuracy'].notna().sum()
        if args.query_pareto_hwnas:
            print(f"📊 帕累托前沿统计: {len(pareto_front)} 个解，其中 {hwnas_available} 个有HW-NAS-Bench accuracy数据")
            if hwnas_available > 0:
                valid_accs = pareto_df['HW-NAS-Bench_Accuracy'].dropna()
                print(f"   HW-NAS-Bench accuracy范围: {valid_accs.min()*100:.2f}% - {valid_accs.max()*100:.2f}%")
                print(f"   平均accuracy: {valid_accs.mean()*100:.2f}%")
            logging.info(f"帕累托前沿中有 {hwnas_available}/{len(pareto_front)} 个解有HW-NAS-Bench数据")
        else:
            print(f"📊 帕累托前沿统计: {len(pareto_front)} 个解")
            logging.info(f"保存了 {len(pareto_front)} 个帕累托最优解")

    # 为简单起见，我们选择一个SWAP分数最高的解进行训练
    # 实际项目中可以根据需求选择平衡点
    print("\n=== 选择最佳架构 ===")
    best_individual = best_result  # 这个已经是SWAP最高的解
    
    print(f"🏆 最佳架构信息:")
    print(f"SWAP分数: {best_individual['fitness_values'][0]:.3f}")
    print(f"功耗估计: {best_individual['fitness_values'][1]:.3f} mJ")
    print(f"op_codes: {best_individual['op_codes']}")
    print(f"width_codes: {best_individual['width_codes']}")
    
    logging.info(f"选择SWAP分数最高的架构进行训练:")
    logging.info(f"最佳架构: op_codes={best_individual['op_codes']}, width_codes={best_individual['width_codes']}")
    logging.info(f"SWAP: {best_individual['fitness_values'][0]:.3f}, Power: {best_individual['fitness_values'][1]:.3f}")

    # 打印架构的详细信息，便于理解最终模型
    print("\n=== 架构详细信息 ===")
    op_codes = best_individual['op_codes']
    width_codes = best_individual['width_codes']
    block_names = []
    for i, op_idx in enumerate(op_codes):
        block_names.append(search_space.op_list[op_idx])
    
    stage_widths = []
    for stage_idx, width_idx in enumerate(width_codes):
        stage_widths.append(f"{search_space.width_choices[width_idx]:.2f}")
    
    print(f"操作类型: {block_names}")
    print(f"各阶段宽度: {stage_widths}")
    
    logging.info(f"架构详情:")
    logging.info(f"  操作类型: {block_names}")
    logging.info(f"  各阶段宽度: {stage_widths}")

    # 8) 构造最优模型
    print("\n=== 构建最优模型 ===")
    
    # 检查网络代码
    print(f"检查架构代码:")
    print(f"  操作码: {best_individual['op_codes']}")
    print(f"  宽度码: {best_individual['width_codes']}")
    
    # 验证代码长度与搜索空间匹配
    if len(best_individual["op_codes"]) != search_space.total_blocks:
        print(f"警告: 操作码长度 ({len(best_individual['op_codes'])}) 与搜索空间的块数 ({search_space.total_blocks}) 不匹配")
    
    if len(best_individual["width_codes"]) != len(search_space.stage_setting):
        print(f"警告: 宽度码长度 ({len(best_individual['width_codes'])}) 与搜索空间的阶段数 ({len(search_space.stage_setting)}) 不匹配")
    
    # 构建前检查搜索空间
    print(f"搜索空间内操作列表: {search_space.op_list}")
    print(f"宽度选项: {search_space.width_choices}")
    
    try:
        best_model = search_space.build_model(best_individual["op_codes"], best_individual["width_codes"])
        param_mb = count_parameters_in_MB(best_model)
        print(f"模型参数量: {param_mb:.2f} MB")
        
        # 验证模型结构
        print("验证模型结构:")
        print(f"输入大小: {'32x32' if search_space.small_input else '224x224'}")
        print(f"输出类别数: {search_space.num_classes}")  # 修正属性名
        
        # 使用更好的初始化
        print("执行改进的模型初始化...")
        for m in best_model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:  # 检查weight是否存在
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 检查bias是否存在
                    nn.init.constant_(m.bias, 0)
                
        # 检查网络层级和结构
        total_layers = 0
        conv_layers = 0
        bn_layers = 0
        linear_layers = 0
        
        for name, module in best_model.named_modules():
            total_layers += 1
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
            elif isinstance(module, nn.BatchNorm2d):
                bn_layers += 1
            elif isinstance(module, nn.Linear):
                linear_layers += 1
                
        print(f"网络结构统计: 总层数={total_layers}, 卷积层={conv_layers}, BN层={bn_layers}, 全连接层={linear_layers}")
        
        # 尝试进行一次前向传播以检查网络
        try:
            print("执行测试性前向传播...")
            dummy_input = torch.randn(1, 3, 32 if search_space.small_input else 224, 32 if search_space.small_input else 224)
            dummy_input = dummy_input.to(device)
            best_model = best_model.to(device)
            
            with torch.no_grad():
                output = best_model(dummy_input)
            
            print(f"前向传播成功! 输出形状: {output.shape}")
        except Exception as e:
            print(f"前向传播测试失败: {e}")
    
    except Exception as e:
        print(f"构建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    logging.info(f"最佳模型参数量: {param_mb:.2f} MB")

    # 9) 尝试从HW-NAS-Bench查询accuracy
    print("\n=== 查询HW-NAS-Bench中的accuracy ===")
    hwnas_accuracy = None
    hwnas_found = False
    
    # 根据数据集选择合适的查询名称
    hwnas_dataset = args.dataset
    if args.dataset == "cifar10":
        hwnas_dataset = "cifar10"
    elif args.dataset == "cifar100":
        hwnas_dataset = "cifar100"
    elif args.dataset == "imagenet":
        hwnas_dataset = "imagenet"
    
    try:
        print(f"正在从HW-NAS-Bench查询 {hwnas_dataset} 数据集上的accuracy...")
        hwnas_accuracy, hwnas_found = swap_metric.query_hwnas_accuracy(
            best_individual["op_codes"], 
            best_individual["width_codes"], 
            dataset=hwnas_dataset
        )
        
        if hwnas_found and hwnas_accuracy is not None:
            print(f"🎯 从HW-NAS-Bench查询到accuracy: {hwnas_accuracy*100:.2f}%")
            logging.info(f"HW-NAS-Bench accuracy: {hwnas_accuracy*100:.2f}%")
            
            # 查询所有指标以获得更完整的信息
            print("正在查询完整的HW-NAS-Bench指标...")
            hwnas_metrics, metrics_found = swap_metric.query_hwnas_metrics(
                best_individual["op_codes"], 
                best_individual["width_codes"], 
                dataset=hwnas_dataset
            )
            
            if metrics_found:
                print("📊 HW-NAS-Bench完整指标:")
                for key, value in hwnas_metrics.items():
                    if isinstance(value, (int, float)):
                        if 'acc' in key.lower():
                            print(f"  {key}: {value*100:.2f}%")
                        elif 'energy' in key.lower() or 'power' in key.lower():
                            print(f"  {key}: {value:.3f} mJ")
                        elif 'latency' in key.lower():
                            print(f"  {key}: {value:.3f} ms")
                        else:
                            print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                        
                logging.info(f"HW-NAS-Bench完整指标: {hwnas_metrics}")
        else:
            print("❌ 无法从HW-NAS-Bench查询到accuracy，可能需要进行训练")
            logging.info("无法从HW-NAS-Bench查询到accuracy")
            
    except Exception as e:
        print(f"⚠️ 查询HW-NAS-Bench时出错: {e}")
        logging.warning(f"查询HW-NAS-Bench时出错: {e}")

    # 10) 决定是否需要训练
    should_train = True
    if hwnas_found and hwnas_accuracy is not None:
        print(f"\n=== 发现HW-NAS-Bench accuracy: {hwnas_accuracy*100:.2f}% ===")
        
        if args.skip_training:
            print("用户选择跳过训练，将使用HW-NAS-Bench的accuracy作为最终结果")
            should_train = False
        else:
            # 询问用户是否还要训练（在自动化脚本中可以设置默认行为）
            print("选项:")
            print("1. 使用HW-NAS-Bench的accuracy，跳过训练（推荐）")
            print("2. 仍然进行训练以验证结果")
            
            # 在自动化环境中，如果HW-NAS-Bench accuracy较高，可以选择跳过训练
            if hwnas_accuracy > 0.7:  # 如果accuracy超过70%，则跳过训练
                print(f"accuracy较高（{hwnas_accuracy*100:.2f}%），自动选择跳过训练")
                should_train = False
            else:
                print(f"accuracy较低（{hwnas_accuracy*100:.2f}%），将进行训练以获得更好结果")
    else:
        # 未在HW-NAS-Bench中找到accuracy，必须进行训练
        print("\n=== 未找到HW-NAS-Bench accuracy，将执行完整训练 ===")
        should_train = True
        # 即使用户设置了skip_training，也需要训练
        if args.skip_training:
            print("注意: 虽然设置了skip_training，但由于无法从HW-NAS-Bench获取accuracy，仍将执行训练")
        
        # 如果训练轮数太少，自动增加以确保充分训练
        if args.train_epochs < 50:
            original_epochs = args.train_epochs
            args.train_epochs = max(args.train_epochs, 50)  # 确保至少50轮
            print(f"自动调整训练轮数: {original_epochs} -> {args.train_epochs}")
    
    # 如果不进行完整训练，直接跳过训练步骤
    if args.train_epochs == 0 or (args.skip_training and should_train == False):
        if not should_train:
            print("使用HW-NAS-Bench的accuracy，跳过训练步骤")
        else:
            print("跳过训练步骤")
        print(f"搜索结果已保存到日志: {log_path}")
        
        # 保存搜索结果，包含HW-NAS-Bench的accuracy
        model_path = os.path.join(log_path, f"best_model_swap_power_{args.dataset}.pth")
        save_dict = {
            'op_codes': best_individual['op_codes'],
            'width_codes': best_individual['width_codes'],
            'state_dict': best_model.state_dict(),
            'swap_score': best_individual['fitness_values'][0],
            'power': best_individual['fitness_values'][1],
            'trained': should_train,  # 添加标志指示是否进行了训练
            'train_epochs': args.train_epochs,
        }
        
        if hwnas_found and hwnas_accuracy is not None:
            save_dict['hwnas_accuracy'] = hwnas_accuracy
            save_dict['top1_acc'] = hwnas_accuracy  # 使用HW-NAS-Bench的accuracy
            print(f"✅ 最终accuracy（来自HW-NAS-Bench）: {hwnas_accuracy*100:.2f}%")
            logging.info(f"最终accuracy（来自HW-NAS-Bench）: {hwnas_accuracy*100:.2f}%")
        else:
            save_dict['top1_acc'] = 0.0  # 没有训练，准确率为0
        
        torch.save(save_dict, model_path)
        print(f"模型架构已保存到: {model_path}")
        
        # 可视化帕累托前沿
        if args.visualize and HAS_MATPLOTLIB:
            print("生成帕累托前沿可视化...")
            try:
                viz_path = os.path.join(log_path, f"pareto_front_{args.dataset}.png")
                plot_pareto_front(pareto_front, save_path=viz_path)
                print(f"可视化图已保存到: {viz_path}")
                logging.info(f"帕累托前沿可视化已保存到 {viz_path}")
            except Exception as e:
                print(f"生成可视化图失败: {e}")
                logging.error(f"生成可视化图失败: {e}")
        
        print(f"\n🎉 架构搜索完成！结果保存在: {log_path}")
        return

    # 11) 最终训练
    print(f"\n=== 开始训练最优模型 ({args.train_epochs} epochs) ===")
    if hwnas_found and hwnas_accuracy is not None:
        print(f"HW-NAS-Bench预测accuracy: {hwnas_accuracy*100:.2f}%，将通过训练验证")
    else:
        print(f"未找到HW-NAS-Bench预测，将执行完整训练过程")
        
    # 在未找到HW-NAS-Bench accuracy时，使用更高的初始学习率以加快收敛
    effective_lr = args.lr
    if not hwnas_found or hwnas_accuracy is None:
        effective_lr = max(args.lr, 0.025)  # 确保学习率足够高
        if effective_lr != args.lr:
            print(f"调整学习率: {args.lr} -> {effective_lr}")
    
    final_top1 = train_and_eval(
        best_model,
        train_loader,
        test_loader,
        device=device,
        epochs=args.train_epochs,
        lr=effective_lr,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing
    )
    print(f"✅ 训练完成！最终测试准确率: {final_top1*100:.2f}%")
    logging.info(f"最佳模型的最终测试准确率 (Top-1): {final_top1*100:.2f}%")
    
    # 如果有HW-NAS-Bench的结果，进行对比
    if hwnas_found and hwnas_accuracy is not None:
        diff = abs(final_top1 - hwnas_accuracy) * 100
        print(f"📊 准确率对比:")
        print(f"  HW-NAS-Bench: {hwnas_accuracy*100:.2f}%")
        print(f"  实际训练:     {final_top1*100:.2f}%")
        print(f"  差异:         {diff:.2f}%")
        logging.info(f"准确率对比 - HW-NAS-Bench: {hwnas_accuracy*100:.2f}%, 实际训练: {final_top1*100:.2f}%, 差异: {diff:.2f}%")

    # 保存模型
    model_path = os.path.join(log_path, f"best_model_swap_power_{args.dataset}.pth")
    save_dict = {
        'op_codes': best_individual['op_codes'],
        'width_codes': best_individual['width_codes'],
        'state_dict': best_model.state_dict(),
        'top1_acc': final_top1,
        'swap_score': best_individual['fitness_values'][0],
        'power': best_individual['fitness_values'][1],
        'trained': should_train,  # 添加标志指示是否进行了训练
        'train_epochs': args.train_epochs,
    }
    
    if hwnas_found and hwnas_accuracy is not None:
        save_dict['hwnas_accuracy'] = hwnas_accuracy
        # 如果同时有HW-NAS-Bench预测和实际训练结果，添加对比信息
        if should_train:
            diff = abs(final_top1 - hwnas_accuracy) * 100
            save_dict['accuracy_diff'] = diff
            save_dict['accuracy_diff_percent'] = (final_top1 / hwnas_accuracy - 1) * 100 if hwnas_accuracy > 0 else 0
            
            # 记录对比情况
            comparison_file = os.path.join(log_path, f"hwnas_comparison_{args.dataset}.csv")
            try:
                if not os.path.exists(comparison_file):
                    # 创建新文件并添加表头
                    with open(comparison_file, 'w') as f:
                        f.write("Architecture,SWAP,Power,HW-NAS-Bench_Acc,Trained_Acc,Diff,Diff_Percent\n")
                
                # 附加结果
                with open(comparison_file, 'a') as f:
                    arch_id = f"{best_individual['op_codes']}_{best_individual['width_codes']}"
                    f.write(f"{arch_id},{best_individual['fitness_values'][0]:.2f},{best_individual['fitness_values'][1]:.2f},{hwnas_accuracy*100:.2f},{final_top1*100:.2f},{diff:.2f},{save_dict['accuracy_diff_percent']:.2f}\n")
                
                print(f"HW-NAS-Bench与训练结果对比已保存到: {comparison_file}")
            except Exception as e:
                print(f"保存对比结果失败: {e}")
    
    torch.save(save_dict, model_path)
    print(f"模型已保存到: {model_path}")
    logging.info(f"模型已保存到 {model_path}")

    # 可视化帕累托前沿
    if args.visualize and HAS_MATPLOTLIB:
        print("生成帕累托前沿可视化...")
        try:
            viz_path = os.path.join(log_path, f"pareto_front_{args.dataset}.png")
            plot_pareto_front(pareto_front, save_path=viz_path)
            print(f"可视化图已保存到: {viz_path}")
            logging.info(f"帕累托前沿可视化已保存到 {viz_path}")
        except Exception as e:
            print(f"生成可视化图失败: {e}")
            logging.error(f"生成可视化图失败: {e}")
    
    print(f"\n🎉 所有任务完成！结果保存在: {log_path}")


if __name__ == "__main__":
    main() 