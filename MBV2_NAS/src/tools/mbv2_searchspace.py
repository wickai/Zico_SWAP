import torch
import torch.nn as nn
import random
import logging
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
                layers.append(self._op_factory(
                    op_name, in_c, out_c, stride, t))

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
        [6, 24, 2, 2],
        [6, 32, 3, 2],
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
