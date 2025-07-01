
import torch
from torchvision import datasets, transforms


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

    train_ds = datasets.CIFAR10(
        root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(
        root, train=False, download=True, transform=transform_test)

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
