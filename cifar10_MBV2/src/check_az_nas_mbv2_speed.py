import argparse
import torch
from torch import nn
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import time

from test_az_nas import compute_az_nas_score
from mobilenetv2 import mobilenet_v2


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


inter_feats = []


def set_inplace_false(model: nn.Module):
    for module in model.modules():
        if hasattr(module, 'inplace'):
            try:
                module.inplace = False
                print(f"Set inplace=False for {module.__class__.__name__}")
            except Exception as e:
                print(f"Failed to set inplace for {module}: {e}")

# def _hook_fn(module, inp, out):
#     feats = out  # .detach()  # .reshape(out.size(0), -1)
#     print('',out.grad_fn)
#     inter_feats.append(feats)


def _make_hook(name):
    def _hook_fn(module, inp, out):
        print(f"[{name}] grad_fn: {out.grad_fn}")
        out.retain_grad()
        inter_feats.append(out)
        if torch.isnan(out).any():
            print(f"[{name}] ⚠️ contains NaN in forward output!")
        if torch.isinf(out).any():
            print(f"[{name}] ⚠️ contains Inf in forward output!")
    return _hook_fn


def main():
    global inter_feats
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

    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             [0.2470, 0.2435, 0.2616]),
    ])

    # 加载CIFAR10数据集
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = mobilenet_v2(num_classes=10, width_mult=1.0, input_size=32,
                         inverted_residual_setting=None)
    # model = MyNet()

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            # h = module.register_forward_hook(_hook_fn)
            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            print(f"{name} inplace={module.inplace}")

    device = torch.device("cuda" if not args.no_cuda else "cpu")

    model.to(device)
    set_inplace_false(model)
    compute_az_nas_score.init_model(model, 'kaiming_norm_fanin')
    model.train()
    inputs = next(iter(train_loader))[0].to(device)
    # with torch.no_grad():
    lossfunc = nn.CrossEntropyLoss().cuda()
    with torch.autograd.detect_anomaly():
        model(inputs)
        print("global #inter_feats", len(inter_feats))
        print("global inter_feats[0].shape", inter_feats[0].shape)
        print("global inter_feats[0].grad_fn", inter_feats[0].grad_fn)

        s_time = time.time()
        gpu = 0  # torch.device("cuda" if use_cuda else "cpu")
        info = compute_az_nas_score.compute_nas_score(model, gpu=gpu, input_=inputs,
                                                      resolution=32,
                                                      batch_size=args.batch_size,
                                                      layer_features=inter_feats)
        print("time:", time.time() - s_time)

        #

    print(info)


if __name__ == '__main__':
    main()
