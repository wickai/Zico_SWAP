import torch
import numpy as np

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
