import argparse
import torch
from torch import nn
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
import time

from test_az_nas import compute_az_nas_score
from mobilenetv2 import mobilenet_v2

torch.manual_seed(0)

# sawp related code


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
        self.activations = feats.sign()  # .to(self.device)
        # print("self.activations.shape", self.activations.shape)

    @torch.no_grad()
    def calc_swap(self, reg_factor=1.0):
        if self.activations is None:
            return 0
        # 转置后 unique(dim=0)
        self.activations = self.activations.t()  # => (features, N)
        unique_patterns = torch.unique(self.activations, dim=0).size(0)
        return unique_patterns/self.activations.size(0) * 100  # * reg_factor


##

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
    # set_inplace_false(model)
    compute_az_nas_score.init_model(model, 'kaiming_norm_fanin')
    model.eval()
    inputs = next(iter(train_loader))[0].to(device)
    # with torch.no_grad():
    # lossfunc = nn.CrossEntropyLoss().cuda()

    swap_evaluator = SampleWiseActivationPatterns(device)
    with torch.autograd.detect_anomaly():
        model(inputs)
        print("global #inter_feats", len(inter_feats))
        print("global inter_feats[0].shape", inter_feats[0].shape)
        print("global inter_feats[0].grad_fn", inter_feats[0].grad_fn)

        # gpu = 0  # torch.device("cuda" if use_cuda else "cpu")
        # info = compute_az_nas_score.compute_nas_score(model, gpu=gpu, input_=inputs,
        #                                               resolution=32,
        #                                               batch_size=args.batch_size,
        #                                               layer_features=inter_feats)
        reg_factor = 1.0
        inter_feats = [fea.detach().reshape(fea.size(0), -1)
                       for fea in inter_feats]
        all_feats = torch.cat(inter_feats, dim=1)  # (N, sum_of_features)
        print("all_feats.shape", all_feats.shape)
        s_time = time.time()
        swap_evaluator.collect_activations(all_feats)
        total_swap_score = swap_evaluator.calc_swap(reg_factor)
        print("time:", time.time() - s_time)

        s_time = time.time()
        swap_score_list = []
        for fea in inter_feats:
            swap_evaluator.collect_activations(fea)
            swap_score = swap_evaluator.calc_swap(reg_factor)
            swap_score_list.append(swap_score)
        print("seperate time:", time.time() - s_time)
        #

    print('total_swap_score', total_swap_score)
    print('swap_score_list', swap_score_list)
    print("sum of swap_score_list", np.mean(swap_score_list))


if __name__ == '__main__':
    main()
