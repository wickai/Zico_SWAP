import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from mobilenetv2 import mobilenet_v2
import torch.nn as nn
import numpy as np
import os
import sys

import logging
# logging.basicConfig(level=logging.INFO)

# from train_and_eval import train_and_eval  # 从01_search_aznas.py导入训练函数


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
        correct_top5 += predicted_top5.eq(targets.view(-1,
                                          1).expand_as(predicted_top5)).sum().item()

        total += targets.size(0)

    if total == 0:
        return 0.0, 0.0

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc

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

# ============ 6) 最终训练 ============


def train_and_eval(model, train_loader, test_loader, device,
                   epochs=50, lr=0.01,
                   mixup_alpha=1.0,
                   label_smoothing=0.1,
                   weight_decay=5e-4):
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
    # logging starting training
    logging.info("Starting training...")
    # 将模型移动到指定设备
    model = model.to(device)

    # 多GPU数据并行
    if torch.cuda.device_count() > 1:
        print(f"=> 使用 {torch.cuda.device_count()} 个GPU进行训练...")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = {'epoch': 0, 'state_dict': model.state_dict(),
                  'best_acc': best_acc}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup处理
            if mixup_alpha > 0.:
                mixed_x, y_a, y_b, lam = mixup_data(
                    inputs, labels, alpha=mixup_alpha)
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
    logging.info(
        f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    return final_top1


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
            logging.FileHandler(os.path.join(
                log_path, "train.log"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():

    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 on CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=4e-5, help='weight decay')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument("--log_path", default="./logs/cifar_mbv2",
                        type=str, help="where to save logs")

    args = parser.parse_args()

    # 根据数据集名称创建日志目录
    log_path = os.path.join(args.log_path, f"base_cifar100")
    setup_logger(log_path)
    logging.info(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             [0.2470, 0.2435, 0.2616]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             [0.2470, 0.2435, 0.2616]),
    ])

    # 加载CIFAR100数据集
    num_classes = 100
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=4)

    # 创建模型
    # inverted_residual_setting = [
    #     # t, c, n, s
    #     [1, 16, 1, 1],
    #     [6, 24, 2, 2],
    #     [6, 32, 3, 2],
    #     [6, 64, 4, 1],  # trun to 1
    #     [6, 96, 3, 1],
    #     [6, 160, 3, 1],  # trun to 1
    #     [6, 320, 1, 1],
    # ]

    # another setting maybe good for cifar100
    # inverted_residual_setting = [
    #     # t, c, n, s
    #     [1, 16, 1, 1],
    #     [6, 24, 2, 1],  # trun to 1
    #     [6, 32, 3, 2],  # trun to 1 # could set to 2
    #     [6, 64, 4, 2],
    #     [6, 96, 3, 1],
    #     [6, 160, 3, 2],
    #     [6, 320, 1, 1],
    # ]

    # another setting
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # trun to 1
        [6, 32, 3, 1],  # trun to 1 # could set to 2
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    model = mobilenet_v2(num_classes=num_classes, width_mult=1.0, input_size=32,
                         inverted_residual_setting=inverted_residual_setting)

    # 训练模型
    train_and_eval(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        mixup_alpha=-1,
        label_smoothing=0.1,
        weight_decay=args.weight_decay
    )


if __name__ == '__main__':
    main()
