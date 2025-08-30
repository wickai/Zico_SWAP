import os
import sys
import time
import logging
import argparse
import json

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from tools.evaluation import set_seed, count_parameters_in_MB, get_model_complexity_info
from tools.mbv2_searchspace import MobileNetSearchSpace
from tools.sawp import SWAP, calculate_mu_sigma
from tools.es import EvolutionarySearch
from tools.loader import get_cifar10_dataloaders
from tools.trainer import train_and_eval
import PIL

from datasets import load_dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        "MobileNetV2 Search with SWAP, then final train with cutout/mixup/label_smoothing/cosine LR")
    parser.add_argument("--log_path", default="./logs",
                        type=str, help="where to save logs")
    parser.add_argument("--data_path", default="./data",
                        type=str, help="CIFAR-10 dataset path")
    parser.add_argument("--device", default="cuda",
                        type=str, help="cpu or cuda")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for reproducibility")

    # Evolutionary search params
    parser.add_argument("--population_size", default=40, type=int,
                        help="population size for evolutionary search")
    parser.add_argument("--mutation_rate", default=0.3,
                        type=float, help="mutation prob")
    parser.add_argument("--n_generations", default=200,
                        type=int, help="generations for ES")
    parser.add_argument("--search_batch", default=64, type=int,
                        help="batchsize for SWAP evaluation (just a small mini-batch)")
    # SWAP + param regular
    parser.add_argument("--use_param_regular", action="store_true", default=False,
                        help="use SWAP with param regular factor")
    logging.info("use_param_regular is False")
    parser.add_argument("--n_samples_mu_sigma", default=1000,
                        type=int, help="for random sampling to calc mu & sigma")
    parser.add_argument("--num_inits", default=1, type=int,
                        help="number of times to random-init the model for averaging SWAP score")

    # final training
    parser.add_argument("--train_batch", default=256,
                        type=int, help="batchsize for final training")  # used to be 128
    parser.add_argument("--train_epochs", default=200,
                        type=int, help="epochs for final training")  # used to be 200
    parser.add_argument("--lr", default=0.04, type=float,
                        help="initial lr for final training")  # used to be 0.01

    parser.add_argument("--small_input", action="store_true",
                        default=False, help="适配CIFAR-10的小输入")
    parser.add_argument("--num_classes", default=1000, type=int, help="类别数")

    # cutout
    parser.add_argument("--use_cutout", action="store_true", default=False,
                        help="enable cutout in data augmentation")
    parser.add_argument("--cutout_length", default=16,
                        type=int, help="cutout length")

    # mixup & label smoothing
    parser.add_argument("--mixup_alpha", default=-1, type=float,
                        help="mixup alpha, if 0 then no mixup")
    parser.add_argument("--label_smoothing", default=0.1,
                        type=float, help="label smoothing factor")
    parser.add_argument("--weight_decay", default=5e-4,
                        type=float, help="weight decay")
    #max_flops_cnn
    parser.add_argument("--max_flops_cnn", default=625,
                        type=float, help="max model size of CNN params in MFLOPs")

    args = parser.parse_args()
    return args


def setup_logger(log_path):
    os.makedirs(log_path, exist_ok=True)

    print("--- Set up basicConfig for logger ---")

    # 清除已有的 handler
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    from datetime import datetime

    # 获取当前日期和时间，格式为 YYMMDD_HH_MM
    timestamp = datetime.now().strftime("%y%m%d_%H_%M")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s INFO: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(
                log_path, f"search_imagenet_{timestamp}.log"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_imagenet_dataset(split: str):
    if split == 'validation':
        split = 'val'
    local_path = os.path.join("/data/wk/kai/data/imagenet_arrow", split)
    print("local_path", local_path)
    if os.path.exists(local_path):
        print(f"[✓] 使用本地数据集: {local_path}")
        hf_dataset = load_from_disk(local_path)
    else:
        print("[⭳] 本地数据集不存在，正在从 Hugging Face Hub 加载...")
        # hf_dataset = load_dataset("imagenet-1k", split=split)
        pass
    
    return hf_dataset

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]

        if isinstance(image, list):
            image = np.array(image)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        print(image.size, image.mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = example.get("label", -1)  # 如果没有 label，可返回 -1 或忽略
        
        return image, label

def main():
    args = parse_args()
    setup_logger(args.log_path)
    args_dict = vars(args)  # 把 Namespace 转换成字典
    logging.info("Args:\n" + json.dumps(args_dict, indent=4))

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
    logging.info(f"[*]MobileNetSearchSpace.op_list: {search_space.op_list}")
    for id, op_name in enumerate(search_space.op_list):
        logging.info(f"{id}: {op_name}")

    # 2) 如果要用SWAP的 param regular，就先随机采样，计算 mu & sigma
    mu, sigma = None, None
    print(f"[*] args.use_param_regular: {args.use_param_regular}")
    if args.use_param_regular:
        logging.info(
            f"Sampling {args.n_samples_mu_sigma} archs to calc mu & sigma for param regular ...")
        mu, sigma = calculate_mu_sigma(
            search_space, n_samples=args.n_samples_mu_sigma)
        logging.info(f"[SWAP param regular] mu={mu:.1f}, sigma={sigma:.1f}")

    # 3) 构造一个小批量数据 (search_batch) 用于SWAP评估
    # transform_eval = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4914, 0.4822, 0.4465],
    #                          [0.2470, 0.2435, 0.2616]),
    # ])
    # train_ds = datasets.CIFAR10(
    #     root=args.data_path, train=True, download=True, transform=transform_eval)

    # 数据增强
    mean = [0.485, 0.456, 0.406]  # for imagenet
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # ImageNet需要指定文件夹路径
    # root = '/home/ubuntu/scratch/kaiwei/unzip_imagenet/ILSVRC/Data/CLS-LOC/'
    # train_ds = datasets.ImageFolder(
    #     os.path.join(root, 'train'),
    #     transform=transform_train
    # )
    hf_dataset = load_imagenet_dataset('train')
    train_ds = HFDatasetWrapper(hf_dataset, transform=transform_train)
    search_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.search_batch, shuffle=True, num_workers=4
    )
    mini_inputs, _ = next(iter(search_loader))
    mini_inputs = mini_inputs.to(device)

    # 4) 构建 SWAP 指标
    print(f"[*] args.use_param_regular, {args.use_param_regular}")
    swap_metric = SWAP(
        device=device, regular=args.use_param_regular, mu=mu, sigma=sigma)

    # 5) 进化搜索
    es = EvolutionarySearch(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        n_generations=args.n_generations,
        swap_metric=swap_metric,
        search_space=search_space,
        device=device,
        num_inits=args.num_inits,
        max_flops_cnn=args.max_flops_cnn
    )
    start = time.time()
    best_individual, population  = es.search(mini_inputs)
    end = time.time()

    logging.info(f"Search finished in {end - start:.2f}s.")
    logging.info(
        f"Best architecture | SWAP fitness={best_individual['fitness']:.3f}")

    # 6) 构造最优模型 & 最终训练
    best_model = search_space.get_model(
        best_individual["op_codes"], best_individual["width_codes"])
    # param_mb = count_parameters_in_MB(best_model)
    # logging.info(f"Best Model param: {param_mb:.2f} MB")
    dummy_input = torch.randn(1,3,224,224)
    model_info = get_model_complexity_info(best_model, dummy_input)
    logging.info(f"Best Model param: {model_info['params']/1e6:.2f} MB, flops: {model_info['flops']/1e6:.2f} MFLOPs")
    logging.info(
        f"Parameters: lr={args.lr}, train_batch={args.train_batch}, train_epochs={args.train_epochs}, mixup_alpha={args.mixup_alpha}, label_smoothing={args.label_smoothing}")
    logging.info(
        f"Best architecture | SWAP fitness={best_individual['fitness']:.3f}")
    logging.info(
        f"Best architecture | op_codes={best_individual['op_codes']}, width_codes={best_individual['width_codes']}")
    
    for i, p in enumerate(population):
        logging.info(f"[{i} item] ============================")
        for k, v in p.items():
            logging.info(f"{k}:{v}")
        logging.info("")
        
        


if __name__ == "__main__":
    main()
