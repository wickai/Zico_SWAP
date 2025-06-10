import torch
import torch.nn as nn

# 设置随机种子
torch.manual_seed(0)

# 保存中间激活输出（ReLU的输出）
inter_feats = []

# hook函数模板，用于捕获中间特征


def _make_hook(name):
    def _hook_fn(module, inp, out):
        print(f"[{name}] grad_fn: {out.grad_fn}")
        out.retain_grad()  # 为后续 autograd.grad 计算梯度保留计算图
        inter_feats.append(out)
    return _hook_fn

# 网络定义


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)


# 输入：10张 3通道 32x32 图片
x = torch.randn(10, 3, 32, 32, requires_grad=True).to("cuda")

# 模型实例
model = MyNet()
# x
model.to("cuda")

# 注册 hook：对所有 ReLU 或 ReLU6 模块
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (nn.ReLU, nn.ReLU6)):
        h = module.register_forward_hook(_make_hook(name))
        hooks.append(h)

# 前向传播
output = model(x)


# grad_output = torch.ones_like(output)

# 从 inter_feats 中获取 ReLU 输出
last_relu_input = inter_feats[-3]
relu_output = inter_feats[-2]

# 手动构造一个 loss 梯度（这里用全 1）
g_out = torch.ones_like(relu_output) * 0.5
g_out = (torch.bernoulli(g_out) - 0.5) * 2

# 手动计算 ReLU 输出对它的输入（也就是ReLU输入 = BN输出）的梯度
grad_relu_input = torch.autograd.grad(
    outputs=relu_output,
    inputs=last_relu_input,
    grad_outputs=g_out,
    retain_graph=False,
    # create_graph=False
)[0]

print("grad_relu_input.shape:", grad_relu_input.shape)  # 应为 [10, 8, 32, 32]
print("grad_relu_input", grad_relu_input)

# 清理 hook（释放资源）
for h in hooks:
    h.remove()
