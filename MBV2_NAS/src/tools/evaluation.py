import random
import numpy as np
import torch

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
