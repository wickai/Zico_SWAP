#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于SWAP (Sample-Wise Activation Patterns) 的神经网络架构搜索实现
使用searchspace_AZNAS_ImageNet中的MobileNetV2搜索空间，支持多种数据集
"""
import os
import sys
import time
import math
import random
import logging
import argparse
import numpy as np
import pandas as pd

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

# ============ 1) 模型评估函数 ============

@torch.no_grad()
def evaluate(model, loader, device):
    """计算Top-1和Top-5准确率"""
    # 确保模型处于评估模式
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # Top-1准确率
        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(targets).sum().item()
        
        # Top-5准确率 (对于CIFAR-100更有意义)
        _, predicted_top5 = outputs.topk(5, 1, True, True)
        correct_top5 += predicted_top5.eq(targets.view(-1, 1).expand_as(predicted_top5)).sum().item()
        
        total += targets.size(0)
    
    if total == 0:
        return 0.0, 0.0

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
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
    val = -((param_kb - mu) ** 2) / (2.0 * (sigma ** 2))
    #val = -((param_kb - mu) ** 2) / sigma
    #val = (param_kb - mu) / (2 * sigma)
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
    """
    def __init__(self, device, regular=True, mu=None, sigma=None):
        self.device = device
        self.regular = regular
        self.mu = mu
        self.sigma = sigma

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
    AZNAS风格进化算法: init population -> 评估 -> 排序 -> 选择 + 交叉 + 变异 -> 重复
    用 SWAP.evaluate(...) 作为适应度 (fitness).
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
        
    def tournament_selection(self, population):
        """锦标赛选择，增加多样性考虑"""
        selected = []
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体
            candidates = random.sample(population, self.tournament_size)
            
            # 有时随机选择以增加多样性
            if random.random() < self.diversity_weight:
                best = random.choice(candidates)
            else:
                # 大部分时间选择适应度最高的
                best = max(candidates, key=lambda x: x["fitness"])
            
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
                    "fitness": None
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
                    if indiv["fitness"] is None:
                        scores = []
                        for _ in range(self.num_inits):
                            model = self.search_space.build_model(indiv["op_codes"], indiv["width_codes"]).to(self.device)
                            # 简单初始化
                            for p in model.parameters():
                                if p.dim() > 1:
                                    nn.init.kaiming_normal_(p)
                            score = self.swap_metric.evaluate(model, inputs)
                            scores.append(score)
                        indiv["fitness"] = float(np.mean(scores))

                    logging.info(f"    [Ind-{i+1}] op_codes: {indiv['op_codes']}, width_codes: {indiv['width_codes']} | fitness(SWAP): {indiv['fitness']:.3f}")

                # 排序 (从大到小)
                population.sort(key=lambda x: x["fitness"], reverse=True)
                logging.info(f"    Best in SubPop{subpop_idx+1}: op_codes={population[0]['op_codes']}, width_codes={population[0]['width_codes']}, fit={population[0]['fitness']:.3f}")
                
                if gen < self.n_generations - 1:  # 最后一代不需要更新
                    # 使用锦标赛选择
                    selected = self.tournament_selection(population)
                    
                    # 创建新一代
                    next_gen = []
                    
                    # 保留精英
                    next_gen.append(population[0])
                    
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
                            "fitness": None
                        })
                    
                    # 更新种群
                    subpops[subpop_idx] = next_gen
            
            # 子群体之间的交流（每5代）
            if gen % 5 == 0 and gen > 0 and gen < self.n_generations - 1:
                logging.info("  执行子群体间交流...")
                for i in range(n_subpops):
                    next_i = (i + 1) % n_subpops
                    # 交换每个子群体的最佳个体
                    subpops[i].append(subpops[next_i][0])
                    subpops[next_i].append(subpops[i][0])
                    # 移除较差个体以保持规模
                    subpops[i] = sorted(subpops[i], key=lambda x: x["fitness"] if x["fitness"] is not None else -1, reverse=True)[:len(subpops[i])-1]
                    subpops[next_i] = sorted(subpops[next_i], key=lambda x: x["fitness"] if x["fitness"] is not None else -1, reverse=True)[:len(subpops[next_i])-1]

        # 最后一代再做一遍评估，确保所有个体都有评估值
        all_individuals = []
        for population in subpops:
            for indiv in population:
                if indiv["fitness"] is None:
                    scores = []
                    for _ in range(self.num_inits):
                        model = self.search_space.build_model(indiv["op_codes"], indiv["width_codes"]).to(self.device)
                        for p in model.parameters():
                            if p.dim() > 1:
                                nn.init.kaiming_normal_(p)
                        score = self.swap_metric.evaluate(model, inputs)
                        scores.append(score)
                    indiv["fitness"] = float(np.mean(scores))
                all_individuals.append(indiv)

        # 从所有子群体中选择最佳个体
        all_individuals.sort(key=lambda x: x["fitness"], reverse=True)
        best = all_individuals[0]
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

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = {'epoch': 0, 'state_dict': model.state_dict(), 'best_acc': best_acc}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
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
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            # 仅用于查看train top1（对mixup只是近似统计）
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc_top1 = correct_top1 / total if total > 0 else 0.

        # 在测试集上计算Top-1 / Top-5
        test_top1, test_top5 = evaluate(model, test_loader, device)
        
        # 保存最佳模型
        if test_top1 > best_acc:
            best_acc = test_top1
            best_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }

        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Loss={train_loss:.3f}, "
                     f"Train@1={train_acc_top1*100:.2f}%, "
                     f"Test@1={test_top1*100:.2f}%, Test@5={test_top5*100:.2f}%")

    # 恢复最佳模型
    model.load_state_dict(best_state['state_dict'])
    final_top1, final_top5 = evaluate(model, test_loader, device)
    logging.info(f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
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
    
    # 训练集数据增强
    if dataset_name in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
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
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
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

# ============ 10) 主函数入口 ============

def parse_args():
    parser = argparse.ArgumentParser("MobileNetV2 Architecture Search with SWAP")
    parser.add_argument("--dataset", type=str, default="cifar100", 
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help="选择数据集: cifar10, cifar100, imagenet")
    parser.add_argument("--log_path", default="./logs/swap_search", type=str, help="where to save logs")
    parser.add_argument("--data_path", default="./data", type=str, help="数据集路径")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")

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

    # cutout
    parser.add_argument("--use_cutout", action="store_true", help="enable cutout in data augmentation")
    parser.add_argument("--cutout_length", default=16, type=int, help="cutout length")

    # mixup & label smoothing
    parser.add_argument("--mixup_alpha", default=0.8, type=float, help="mixup alpha, if 0 then no mixup")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing factor")

    # 添加锦标赛大小参数
    parser.add_argument("--tournament_size", default=3, type=int, 
                       help="tournament size for selection in evolutionary search")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 根据数据集名称创建日志目录
    log_path = os.path.join(args.log_path, f"swap_{args.dataset}")
    setup_logger(log_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    logging.info(f"使用数据集: {args.dataset}")

    # 1) 加载数据集并获取配置信息
    train_loader, test_loader, mean, std, num_classes, small_input = get_dataset_dataloaders(
        dataset_name=args.dataset,
        root=args.data_path,
        batch_size=args.train_batch,
        num_workers=4,
        use_cutout=args.use_cutout,
        cutout_length=args.cutout_length
    )
    
    # 2) 创建与数据集匹配的搜索空间
    search_space = MobileNetSearchSpace(num_classes=num_classes, small_input=small_input)
    logging.info(f"搜索空间创建完成，使用MobileNetV2架构，针对{args.dataset}数据集")
    
    # 输出搜索空间信息
    logging.info(f"搜索空间: 共 {search_space.total_blocks} 个块, 宽度选项: {search_space.width_choices}")
    logging.info(f"操作类型: {search_space.op_list}")
    logging.info(f"网络结构: {search_space.stage_setting}")
    logging.info(f"类别数: {num_classes}, 小输入: {small_input}")

    # 3) 如果要用SWAP的 param regular，就先随机采样，计算 mu & sigma
    mu, sigma = None, None
    if args.use_param_regular:
        logging.info(f"采样 {args.n_samples_mu_sigma} 个架构来计算参数正则化的 mu & sigma ...")
        mu, sigma = calculate_mu_sigma(search_space, n_samples=args.n_samples_mu_sigma)
        logging.info(f"[SWAP param regular] mu={mu:.1f}, sigma={sigma:.1f}")

    # 4) 构造一个小批量数据 (search_batch) 用于SWAP评估
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

    # 5) 构建 SWAP 指标
    swap_metric = SWAP(device=device, regular=args.use_param_regular, mu=mu, sigma=sigma)

    # 6) 进化搜索
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
    best_individual = es.search(mini_inputs)
    end = time.time()

    logging.info(f"搜索完成，耗时 {end - start:.2f}秒.")
    logging.info(f"最佳架构: op_codes={best_individual['op_codes']}, width_codes={best_individual['width_codes']} | SWAP fitness={best_individual['fitness']:.3f}")

    # 打印架构的详细信息，便于理解最终模型
    op_codes = best_individual['op_codes']
    width_codes = best_individual['width_codes']
    block_names = []
    for i, op_idx in enumerate(op_codes):
        block_names.append(search_space.op_list[op_idx])
    
    stage_widths = []
    for stage_idx, width_idx in enumerate(width_codes):
        stage_widths.append(f"{search_space.width_choices[width_idx]:.2f}")
    
    logging.info(f"架构详情:")
    logging.info(f"  操作类型: {block_names}")
    logging.info(f"  各阶段宽度: {stage_widths}")

    # 7) 构造最优模型
    best_model = search_space.build_model(best_individual["op_codes"], best_individual["width_codes"])
    param_mb = count_parameters_in_MB(best_model)
    logging.info(f"最佳模型参数量: {param_mb:.2f} MB")

    # 8) 最终训练
    final_top1 = train_and_eval(
        best_model,
        train_loader,
        test_loader,
        device=device,
        epochs=args.train_epochs,
        lr=args.lr,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing
    )
    logging.info(f"最佳模型的最终测试准确率 (Top-1): {final_top1*100:.2f}%")

    # 保存模型
    torch.save({
        'op_codes': best_individual['op_codes'],
        'width_codes': best_individual['width_codes'],
        'state_dict': best_model.state_dict(),
        'top1_acc': final_top1
    }, os.path.join(log_path, f"best_model_swap_{args.dataset}.pth"))
    logging.info(f"模型已保存到 {os.path.join(log_path, f'best_model_swap_{args.dataset}.pth')}")


if __name__ == "__main__":
    main() 