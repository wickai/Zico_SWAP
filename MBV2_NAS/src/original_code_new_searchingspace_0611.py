import os
import sys
import time
import math
import random
import logging
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np


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
        _, pred_topk = outputs.topk(5, dim=1, largest=True, sorted=True)  # shape: [B, 5]

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


# ============ 2) Cutout自定义数据增强 (仅最终训练使用) ============

class Cutout(object):
    """
    在图像上随机抹去 n_holes 个 length x length 的正方形区域。
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        img: PIL图像 or torch tensor, 大小[C,H,W]
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


# ============ 3) Mixup工具函数 (仅最终训练使用) ============

def mixup_data(x, y, alpha=1.0):
    """
    返回混合后的输入和标签：mixed_x, y_a, y_b, lam
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
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
    print("param_mb, mu, sigma =", param_mb, mu, sigma)
    
    # 建议引入 2.0 以更符合高斯分布
    #val = -((param_kb - mu) ** 2) / (2.0 * (sigma ** 2))
    val = -((param_kb - mu) ** 2) / sigma
    return math.exp(val)


class SampleWiseActivationPatterns:
    """
    跟原先一样的 SWAP 逻辑:
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
        print("unique_patterns * reg_factor:", unique_patterns, reg_factor)
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


# ============ 5) MobileNet相关: SEBlock / Zero / MBConv / MobileNetV2 ============

class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, *_ = x.shape
        y = self.fc(self.avg(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class Zero(nn.Module):
    def __init__(self, stride, out_c):
        super().__init__()
        self.stride, self.out_c = stride, out_c

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if x.size(1) != self.out_c:
            pad = self.out_c - x.size(1)
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        return x.mul(0.0)


class MBConv(nn.Module):
    def __init__(self, inp, oup, k, s, expand, se=False):
        super().__init__()
        self.use_res = (s == 1 and inp == oup)
        hid = inp * expand

        layers = []
        if expand != 1:
            layers += [
                nn.Conv2d(inp, hid, 1, bias=False),
                nn.BatchNorm2d(hid),
                nn.ReLU6(inplace=True)
            ]

        layers += [
            nn.Conv2d(hid, hid, k, s, k // 2, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True)
        ]

        if se:
            layers.append(SEBlock(hid))

        layers += [
            nn.Conv2d(hid, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    """
    按照searchspace.py中的MobileNetV2Proxy实现
    """
    def __init__(self, op_codes, width_codes, stage_setting, op_list, width_choices,
                 num_classes=10, small_input=True):
        super().__init__()
        self._build_ops(op_list)
        
        # Stem
        stem_c = 16
        stem_stride = 1 if small_input else 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, 3, stem_stride, 1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU6(inplace=True),
        )
        
        # Features
        layers = []
        in_c, blk_idx = stem_c, 0
        for stage_idx, (t, c_base, n, s) in enumerate(stage_setting):
            out_c = int(round(c_base * width_choices[width_codes[stage_idx]]))
            for i in range(n):
                stride = s if i == 0 else 1
                
                op_name = op_list[op_codes[blk_idx]]
                layers.append(self._op_factory(op_name, in_c, out_c, stride, t))
                
                in_c, blk_idx = out_c, blk_idx + 1
        self.features = nn.Sequential(*layers)
        
        # Head
        last_c = 1280 if not small_input else 1024  # 小分辨率可降维
        self.head = nn.Sequential(
            nn.Conv2d(in_c, last_c, 1, bias=False),
            nn.BatchNorm2d(last_c),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_c, num_classes)
        
    def _build_ops(self, op_list):
        def mb(k, r, se=False):
            return lambda i, o, s, t: MBConv(i, o, k, s, r, se)
        
        self._ops = {}
        for k in (3, 5, 7):
            for r in (3, 6):
                base = f"mbconv_{k}x{k}_r{r}"
                self._ops[base] = mb(k, r, se=False)
                self._ops[base + "_se"] = mb(k, r, se=True)
        
        self._ops["skip_connect"] = (
            lambda i, o, s, t: nn.Identity() if s == 1 and i == o else Zero(s, o)
        )
        self._ops["zero"] = lambda i, o, s, t: Zero(s, o)
        self.op_list = list(op_list)
        
    def _op_factory(self, name, i, o, s, t):
        return self._ops[name](i, o, s, t)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetSearchSpace:
    """
    按照searchspace.py实现的搜索空间
    """
    _STAGE_SETTING = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # turn to 1
        [6, 32, 3, 1],  # turn to 1 
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    _WIDTH_CHOICES = [1.0, 1.2]
    
    @staticmethod
    def _default_op_list():
        ops = []
        for k in (3, 5, 7):
            for r in (3, 6):
                ops += [f"mbconv_{k}x{k}_r{r}", f"mbconv_{k}x{k}_r{r}_se"]
        ops += ["skip_connect", "zero"]
        return ops
    
    def __init__(self, num_classes=10, small_input=True):
        self.stage_setting = self._STAGE_SETTING
        self.op_list = self._default_op_list()
        self.width_choices = self._WIDTH_CHOICES
        self.total_blocks = sum(s[2] for s in self.stage_setting)
        self.num_classes = num_classes
        self.small_input = small_input
    
    def random_op_codes(self):
        return [random.randrange(len(self.op_list)) for _ in range(self.total_blocks)]
    
    def random_width_codes(self):
        return [random.randrange(len(self.width_choices)) for _ in range(len(self.stage_setting))]
    
    def mutate_op_codes(self, codes, mutation_rate=0.3):
        mutated = codes[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.randrange(len(self.op_list))
        return mutated
        
    def mutate_width_codes(self, codes, mutation_rate=0.3):
        mutated = codes[:]
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.randrange(len(self.width_choices))
        return mutated
    
    def get_model(self, op_codes, width_codes):
        return MobileNetV2(
            op_codes=op_codes,
            width_codes=width_codes,
            stage_setting=self.stage_setting,
            op_list=self.op_list,
            width_choices=self.width_choices,
            num_classes=self.num_classes,
            small_input=self.small_input
        )


# ============ 6) EvolutionarySearch (SWAP作为评估) ============

class EvolutionarySearch:
    """
    简易进化算法: init population -> 评估 -> 排序 -> 选择 + 交叉 + 变异 -> 重复
    用 SWAP.evaluate(...) 作为适应度 (fitness).
    """
    def __init__(self, population_size, mutation_rate, n_generations,
                 swap_metric, search_space, device,
                 num_inits=1):
        """
        :param num_inits: 对同一个结构多次随机初始化并计算SWAP平均，以减少初始化差异
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.swap_metric = swap_metric
        self.search_space = search_space
        self.device = device
        self.num_inits = num_inits

    def search(self, inputs):
        # 1) 初始化种群
        population = []
        for _ in range(self.population_size):
            op_codes = self.search_space.random_op_codes()
            width_codes = self.search_space.random_width_codes()
            population.append({
                "op_codes": op_codes, 
                "width_codes": width_codes, 
                "fitness": None
            })

        for gen in range(self.n_generations):
            logging.info(f"=== Generation {gen+1} / {self.n_generations} ===")

            # 评估适应度
            for i, indiv in enumerate(population):
                if indiv["fitness"] is None:
                    scores = []
                    for _ in range(self.num_inits):
                        model = self.search_space.get_model(
                            indiv["op_codes"], 
                            indiv["width_codes"]
                        ).to(self.device)
                        # 简单初始化
                        for p in model.parameters():
                            if p.dim() > 1:
                                nn.init.kaiming_normal_(p)
                        score = self.swap_metric.evaluate(model, inputs)
                        scores.append(score)
                    indiv["fitness"] = float(np.mean(scores))

                logging.info(f"  [Ind-{i+1}] fitness(SWAP): {indiv['fitness']:.3f}")

            # 排序 (从大到小)
            population.sort(key=lambda x: x["fitness"], reverse=True)
            logging.info(f"  Best in Gen{gen+1}: fitness={population[0]['fitness']:.3f}")

            # 选择前 half
            next_gen = population[: self.population_size // 2]

            # 交叉 + 变异, 直到恢复到 population_size
            while len(next_gen) < self.population_size:
                p1 = random.choice(next_gen)
                p2 = random.choice(next_gen)
                
                # 交叉
                child_op_codes = self.crossover(p1["op_codes"], p2["op_codes"])
                child_width_codes = self.crossover(p1["width_codes"], p2["width_codes"])
                
                # 变异
                child_op_codes = self.search_space.mutate_op_codes(child_op_codes, self.mutation_rate)
                child_width_codes = self.search_space.mutate_width_codes(child_width_codes, self.mutation_rate)
                
                next_gen.append({
                    "op_codes": child_op_codes,
                    "width_codes": child_width_codes,
                    "fitness": None
                })

            # 下一代
            population = next_gen

        # 最后一代再做一遍评估
        for indiv in population:
            if indiv["fitness"] is None:
                scores = []
                for _ in range(self.num_inits):
                    model = self.search_space.get_model(
                        indiv["op_codes"], 
                        indiv["width_codes"]
                    ).to(self.device)
                    for p in model.parameters():
                        if p.dim() > 1:
                            nn.init.kaiming_normal_(p)
                    score = self.swap_metric.evaluate(model, inputs)
                    scores.append(score)
                indiv["fitness"] = float(np.mean(scores))

        population.sort(key=lambda x: x["fitness"], reverse=True)
        best = population[0]
        return best

    @staticmethod
    def crossover(codes1, codes2):
        length = len(codes1)
        point1 = random.randint(0, length - 1)
        point2 = random.randint(point1, length - 1)
        child = codes1[:point1] + codes2[point1:point2] + codes1[point2:]
        return child


# ============ 7) 最终训练 (启用 Cutout / Mixup / Label Smoothing / Cosine LR) ============

def train_and_eval(model, train_loader, test_loader, device,
                   epochs=50, lr=0.01,
                   mixup_alpha=1.0,
                   label_smoothing=0.1):
    """
    对最终搜索到的结构进行完整训练并评估其在测试集上的Top-1/Top-5准确率。
    - 启用 mixup & label smoothing & cutout(在DataLoader里) & CosineAnnealingLR
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_top1, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # ============ mixup处理 ============
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

        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Loss={train_loss:.3f}, "
                     f"Train@1={train_acc_top1*100:.2f}%, "
                     f"Test@1={test_top1*100:.2f}%, Test@5={test_top5*100:.2f}%")

    final_top1, final_top5 = evaluate(model, test_loader, device)
    logging.info(f"Final Test Accuracy: Top1={final_top1*100:.2f}%, Top5={final_top5*100:.2f}%")
    return final_top1


# ============ 8) CIFAR-10 DataLoader（最终训练用） ============

def get_cifar10_dataloaders(root, batch_size, num_workers=2,
                            use_cutout=False, cutout_length=16):
    """
    构造CIFAR-10 DataLoader，支持cutout
    最终训练用
    """
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if use_cutout:
        transform_list.append(Cutout(n_holes=1, length=cutout_length))
    transform_list.append(
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616])
    )
    transform_train = transforms.Compose(transform_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])

    train_ds = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

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
    return train_loader, test_loader


# ============ 9) 计算 SWAP 正则所需的 mu,sigma 的函数 ============

def calculate_mu_sigma(search_space, n_samples=1000):
    """
    SWAP正则化所需的 mu & sigma
    这里我们取中位数做 mu，标准差做 sigma
    """
    arr = []
    for _ in range(n_samples):
        op_codes = search_space.random_op_codes()
        width_codes = search_space.random_width_codes()
        model = search_space.get_model(op_codes, width_codes)
        param_mb = count_parameters_in_MB(model)
        param_kb = param_mb * 1e3
        arr.append(param_kb)

    # 用 pandas 的 describe()
    series_ = pd.Series(arr)
    desc = series_.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    # 中位数
    mu = desc["50%"]
    # 标准差
    sigma = desc["mean"]
    return mu, sigma


# ============ 10) 主函数入口 ============

def parse_args():
    parser = argparse.ArgumentParser("MobileNetV2 Search with SWAP, then final train with cutout/mixup/label_smoothing/cosine LR")
    parser.add_argument("--log_path", default="./logs", type=str, help="where to save logs")
    parser.add_argument("--data_path", default="./data", type=str, help="CIFAR-10 dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")
    parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")

    # Evolutionary search params
    parser.add_argument("--population_size", default=80, type=int, help="population size for evolutionary search")
    parser.add_argument("--mutation_rate", default=0.3, type=float, help="mutation prob")
    parser.add_argument("--n_generations", default=100, type=int, help="generations for ES")
    parser.add_argument("--search_batch", default=32, type=int,
                        help="batchsize for SWAP evaluation (just a small mini-batch)")
    # SWAP + param regular
    parser.add_argument("--use_param_regular", action="store_true", help="use SWAP with param regular factor")
    logging.info("use_param_regular is False") 
    parser.add_argument("--n_samples_mu_sigma", default=1000, type=int, help="for random sampling to calc mu & sigma")
    parser.add_argument("--num_inits", default=1, type=int,
                        help="number of times to random-init the model for averaging SWAP score")

    # final training
    parser.add_argument("--train_batch", default=128, type=int, help="batchsize for final training")
    parser.add_argument("--train_epochs", default=200, type=int, help="epochs for final training")
    parser.add_argument("--lr", default=0.05, type=float, help="initial lr for final training")

    parser.add_argument("--small_input", action="store_true", default=True, help="适配CIFAR-10的小输入")
    parser.add_argument("--num_classes", default=10, type=int, help="类别数")

    # cutout
    parser.add_argument("--use_cutout", action="store_true", help="enable cutout in data augmentation")
    parser.add_argument("--cutout_length", default=16, type=int, help="cutout length")

    # mixup & label smoothing
    parser.add_argument("--mixup_alpha", default=-1, type=float, help="mixup alpha, if 0 then no mixup")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing factor")

    args = parser.parse_args()
    return args


def setup_logger(log_path):
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_path, "0613_01.log"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    args = parse_args()
    setup_logger(args.log_path)
    
    # 设置随机种子确保可重现性
    set_seed(args.seed)
    logging.info(f"Set random seed to {args.seed} for reproducibility")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1) 定义搜索空间 - 现在使用与searchspace.py一致的搜索空间
    search_space = MobileNetSearchSpace(
        num_classes=args.num_classes,
        small_input=args.small_input
    )

    # 2) 如果要用SWAP的 param regular，就先随机采样，计算 mu & sigma
    mu, sigma = None, None
    if args.use_param_regular:
        logging.info(f"Sampling {args.n_samples_mu_sigma} archs to calc mu & sigma for param regular ...")
        mu, sigma = calculate_mu_sigma(search_space, n_samples=args.n_samples_mu_sigma)
        logging.info(f"[SWAP param regular] mu={mu:.1f}, sigma={sigma:.1f}")
    else:
        logging.info("use_param_regular is False！！！！") 

    # 3) 构造一个小批量数据 (search_batch) 用于SWAP评估
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2470, 0.2435, 0.2616]),
    ])
    train_ds = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_eval)
    search_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.search_batch, shuffle=True, num_workers=8
    )
    mini_inputs, _ = next(iter(search_loader))
    mini_inputs = mini_inputs.to(device)

    # 4) 构建 SWAP 指标
    swap_metric = SWAP(device=device, regular=args.use_param_regular, mu=mu, sigma=sigma)

    # 5) 进化搜索
    es = EvolutionarySearch(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        n_generations=args.n_generations,
        swap_metric=swap_metric,
        search_space=search_space,
        device=device,
        num_inits=args.num_inits
    )
    start = time.time()
    best_individual = es.search(mini_inputs)
    end = time.time()

    logging.info(f"Search finished in {end - start:.2f}s.")
    logging.info(f"Best architecture | SWAP fitness={best_individual['fitness']:.3f}")

    # 6) 构造最优模型 & 最终训练
    best_model = search_space.get_model(best_individual["op_codes"], best_individual["width_codes"])
    param_mb = count_parameters_in_MB(best_model)
    logging.info(f"Best Model param: {param_mb:.2f} MB")
    logging.info(f"Parameters: lr={args.lr}, train_batch={args.train_batch}, train_epochs={args.train_epochs}, mixup_alpha={args.mixup_alpha}, label_smoothing={args.label_smoothing}")
    logging.info(f"Best architecture | SWAP fitness={best_individual['fitness']:.3f}")
    logging.info(f"Best architecture | op_codes={best_individual['op_codes']}, width_codes={best_individual['width_codes']}")

    # 准备最终训练 & 测试 DataLoader（带 cutout）
    train_loader, test_loader = get_cifar10_dataloaders(
        root=args.data_path,
        batch_size=args.train_batch,
        num_workers=2,
        use_cutout=args.use_cutout,
        cutout_length=args.cutout_length
    )

    # 用 mixup + label_smoothing + cos LR 完整训练
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
    logging.info(f"Final Accuracy of Best Model (Top-1): {final_top1*100:.2f}%")

    # 保存最终模型，如需要请解除注释
    # torch.save(best_model.state_dict(), os.path.join(args.log_path, "best_model.pth"))


if __name__ == "__main__":
    main()
