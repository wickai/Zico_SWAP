import random
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, parameter_count_table
import torchprofile
import time
# import logging
import torch.nn as nn

import warnings
# 忽略 torchprofile 里关于 aten::pad 的特定警告
warnings.filterwarnings(
    "ignore",
    message=r'No handlers found: "aten::pad". Skipped.'
)

def get_model_complexity_info(model, inputs):
    start_time = time.time()

    flops = torchprofile.profile_macs(model, inputs)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    end_time = time.time()
    elapsed = end_time - start_time

    return {
        "params": params,
        "flops": flops,
        "time_seconds": elapsed
    }
    
def get_model_complexity_info_fvcore(model, inputs):
    """
    返回模型的参数数量（params）、激活数（acts）、FLOPs（flops），单位为原始数值（非MB、非GFLOPs）。
    Args:
        model: PyTorch 模型
        inputs: 输入样本（如 torch.randn(1, 3, 224, 224)）
    Returns:
        dict, 包含 "params", "acts", "flops"
    """
    start_time = time.time()
    flops = FlopCountAnalysis(model, inputs)
    op_flops = flops.by_operator()
    total_flops = sum(op_flops.values())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    end_time = time.time()

    return {
        "params": num_params,
        "flops": total_flops,
        "flops_cnn": op_flops["conv"] + op_flops["linear"],
        "time_seconds": end_time - start_time
    }
    
# ============ 1) 通用工具函数 ============


def set_seed(seed):
    """
    设置随机种子，确保实验可重现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters_in_MB(model):
    """
    计算模型可训练参数量，单位：MB
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def evaluate(model, loader, device):
    """
    计算Top-1和Top-5的准确率
    返回 (top1, top5)
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)      # shape: [B, 10] for CIFAR-10
        # 取 top5
        _, pred_topk = outputs.topk(
            5, dim=1, largest=True, sorted=True)  # shape: [B, 5]

        # 计算 top1
        correct_top1 += (pred_topk[:, 0] == labels).sum().item()

        # 计算 top5
        for i in range(labels.size(0)):
            if labels[i].item() in pred_topk[i].tolist():
                correct_top5 += 1

        total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc
