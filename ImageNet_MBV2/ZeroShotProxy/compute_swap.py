import torch
from torch import nn
import math
from PlainNet.basic_blocks import RELU
def count_parameters_in_MB(model: nn.Module):
    """
    计算模型参数量（单位：MB）
    """
    # 这里为了示例，只简单返回 float 类型数值，可以根据你的项目需求进行调整
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    return num_params * 4 / (1024.0 * 1024.0)  # 假设float32，每个param占4 bytes

def cal_regular_factor(model, mu, sigma):
    
    param_mb = count_parameters_in_MB(model)
    param_kb = param_mb * 1e3
    print("param_mb, mu, sigma =", param_mb, mu, sigma)
    val = -((param_kb - mu) ** 2) / (2*sigma**2)  
    return math.exp(val)


class SampleWiseActivationPatterns:
    """
    收集所有 ReLU 的特征 sign()，然后求 unique 模式数量
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
        # shape: (N, sum_of_features)
        # 转置后，每一行代表一个feature，每一列代表一个样本
        self.activations = self.activations.t()
        unique_patterns = torch.unique(self.activations, dim=0).size(0)
        print("unique_patterns * reg_factor:", unique_patterns, reg_factor)
        return unique_patterns #* reg_factor


class SWAP:
    """
    SWAP 结合参数量正则

    使用方法：
        1. 初始化 SWAP 实例:
            swap_evaluator = SWAP(device, regular=True, mu=..., sigma=...)
        2. 调用 evaluate:
            swap_score = swap_evaluator.evaluate(model, inputs)
    """
    def __init__(self, device, regular=False, mu=None, sigma=None):
        self.device = device
        self.regular = regular
        self.mu = mu
        self.sigma = sigma
        self.inter_feats = []
        self.swap_evaluator = SampleWiseActivationPatterns(device)

    def evaluate(self, model, inputs):
        """
        给定模型和输入样本，计算SWAP分数.

        参数:
            model: PyTorch 模型
            inputs: 一个 batch 的输入数据，tensor
        返回:
            SWAP 分数 (float)
        """
        # 如果启用了参数量正则
        if self.regular and (self.mu is not None) and (self.sigma is not None):
            reg_factor = cal_regular_factor(model, self.mu, self.sigma)
        else:
            reg_factor = 1.0

        # 注册 hook 收集 ReLU 输出
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, RELU):
                h = module.register_forward_hook(self._hook_fn)
                hooks.append(h)
        print(hooks)
        self.inter_feats = []
        model.eval()
        with torch.no_grad():
            model(inputs.to(self.device))

        if len(self.inter_feats) == 0:
            self._clear_hooks(hooks)
            return 0

        # 拼接所有中间特征： shape (N, sum_of_features)
        all_feats = torch.cat(self.inter_feats, dim=1)
        print("all_feats", all_feats)
        self.swap_evaluator.collect_activations(all_feats)
        
        swap_score = self.swap_evaluator.calc_swap(reg_factor)
        self._clear_hooks(hooks)
        self.inter_feats = []
        return swap_score

    def _hook_fn(self, module, inp, out):
        print(f"Hook triggered for module: {module}")
        # out: shape (N, C, H, W) --> reshape to (N, -1)
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)

    def _clear_hooks(self, hooks):
        for h in hooks:
            h.remove()
        hooks.clear()


def compute_swap_score(gpu, model, resolution, batch_size,
                       regular=False, mu=None, sigma=None):

    # 将模型放到指定GPU(如果gpu is not None)
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # 随机生成一个 batch 的输入
    # 与其他零成本指标一致，这里用随机输入即可
    inputs = torch.randn(size=[batch_size, 3, resolution, resolution])
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    inputs = inputs.to(device)
    print("inputs", inputs)

    # 初始化一个SWAP对象
    swap_evaluator = SWAP(device=device, regular=regular, mu=mu, sigma=sigma)

    # 计算SWAP score
    swap_score = swap_evaluator.evaluate(model, inputs)
    print("swap_score", swap_score)
    return float(swap_score)
