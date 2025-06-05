#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AZ-NAS MobileNetV2 Search Space
--------------------------------
• 21 可搜索 blocks
• 14 操作: 12×MBConv(±SE) + skip_connect + zero
• Stage-wise 宽度 {1.0, 1.2}
• small_input=True 时适配 32×32（CIFAR-10/100）
"""

from __future__ import annotations
import torch
import torch.nn as nn
import random
from typing import List, Sequence

# -------------------------------------------------------------------------- #
# 🔧 工具
def count_parameters_in_MB(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# -------------------------------------------------------------------------- #
# 🧩 基础组件
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 4):
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
    def __init__(self, stride: int, out_c: int):
        super().__init__()
        self.stride, self.out_c = stride, out_c

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if x.size(1) != self.out_c:
            pad = self.out_c - x.size(1)
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        return x.mul(0.0)


class MBConv(nn.Module):
    def __init__(self, inp, oup, k, s, expand, se=False):
        super().__init__()
        self.use_res = (s == 1 and inp == oup)
        hid = inp * expand

        layers: List[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(inp, hid, 1, bias=False),
                       nn.BatchNorm2d(hid),
                       nn.ReLU6(inplace=True)]

        layers += [nn.Conv2d(hid, hid, k, s, k // 2, groups=hid, bias=False),
                   nn.BatchNorm2d(hid),
                   nn.ReLU6(inplace=True)]

        if se:
            layers.append(SEBlock(hid))

        layers += [nn.Conv2d(hid, oup, 1, bias=False),
                   nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


# -------------------------------------------------------------------------- #
# 🏗️ Proxy Model
class MobileNetV2Proxy(nn.Module):
    def __init__(
        self,
        op_codes: Sequence[int],
        width_codes: Sequence[int],
        stage_setting: Sequence[Sequence[int]],
        op_list: List[str],
        width_choices: Sequence[float],
        num_classes: int = 1000,
        small_input: bool = False,
    ):
        super().__init__()
        self._build_ops(op_list)

        # ---- Stem ----
        stem_c = 16
        stem_stride = 1 if small_input else 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_c, 3, stem_stride, 1, bias=False),
            nn.BatchNorm2d(stem_c),
            nn.ReLU6(inplace=True),
        )

        # ---- Features ----
        layers: List[nn.Module] = []
        in_c, blk_idx = stem_c, 0
        for stage_idx, (t, c_base, n, s) in enumerate(stage_setting):
            out_c = int(round(c_base * width_choices[width_codes[stage_idx]]))
            for i in range(n):
                stride = s if i == 0 else 1
                op_name = op_list[op_codes[blk_idx]]
                layers.append(self._op_factory(op_name, in_c, out_c, stride, t))
                in_c, blk_idx = out_c, blk_idx + 1
        self.features = nn.Sequential(*layers)

        # ---- Head ----
        last_c = 1280 if not small_input else 1024  # 小分辨率可降维
        self.head = nn.Sequential(
            nn.Conv2d(in_c, last_c, 1, bias=False),
            nn.BatchNorm2d(last_c),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_c, num_classes)

    # --------------------
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    # --------------------
    def _build_ops(self, op_list: List[str]):
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


# -------------------------------------------------------------------------- #
# 🔍 Search Space
class MobileNetSearchSpace:
    _STAGE_SETTING = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 3, 2],
        [6, 40, 4, 2],
        [6, 80, 4, 2],
        [6, 96, 3, 1],
        [6, 192, 4, 2],
        [6, 320, 2, 1],
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

    # --------------------
    def __init__(self, num_classes=1000, small_input=False):
        self.stage_setting = self._STAGE_SETTING
        self.op_list = self._default_op_list()
        self.width_choices = self._WIDTH_CHOICES
        self.total_blocks = sum(s[2] for s in self.stage_setting)
        self.num_classes = num_classes
        self.small_input = small_input

    # ---------- 随机 / 变异 ----------
    def random_op_codes(self):
        return [random.randrange(len(self.op_list)) for _ in range(self.total_blocks)]

    def random_width_codes(self):
        return [random.randrange(len(self.width_choices)) for _ in range(len(self.stage_setting))]

    # ---------- 构建模型 ----------
    def build_model(self, op_codes, width_codes):
        return MobileNetV2Proxy(
            op_codes,
            width_codes,
            self.stage_setting,
            self.op_list,
            self.width_choices,
            num_classes=self.num_classes,
            small_input=self.small_input,
        )

'''
# -------------------------------------------------------------------------- #
# 🧪 Quick test
if __name__ == "__main__":
    # CIFAR-100 示例：num_classes=100，small_input=True
    space = MobileNetSearchSpace(num_classes=100, small_input=True)
    op_codes = space.random_op_codes()
    width_codes = space.random_width_codes()
    model = space.build_model(op_codes, width_codes)
    x = torch.randn(1, 3, 32, 32)          # CIFAR 输入
    out = model(x)
    print("Output:", out.shape)            # -> [1, 100]
    print("Params (MB):", count_parameters_in_MB(model))

    # ImageNet-1k 示例：num_classes=1000，small_input=False
    space_im = MobileNetSearchSpace(num_classes=1000, small_input=False)
    model_im = space_im.build_model(
        space_im.random_op_codes(), space_im.random_width_codes()
    )
    y = model_im(torch.randn(1, 3, 224, 224))
    print("Output ImageNet:", y.shape)     # -> [1, 1000]
'''