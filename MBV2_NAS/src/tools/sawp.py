import math
import pandas as pd
import torch
import torch.nn as nn
from .evaluation import get_model_complexity_info, count_parameters_in_MB

# ============ 4) SWAP评估相关 ============


def cal_regular_factor(model, mu, sigma):
    """
    基于高斯形惩罚因子:
    factor = exp( -((param_kb - mu)^2) / (2 * sigma^2) )
    返回 float
    """
    param_mb = count_parameters_in_MB(model)
    param_kb = param_mb * 1e3
    

    # 建议引入 2.0 以更符合高斯分布
    # val = -((param_kb - mu) ** 2) / (2.0 * (sigma ** 2))
    val = -((param_kb - mu) ** 2) / sigma
    
    print("param_mb, mu, sigma =", param_mb, mu, sigma)
    ratio = math.exp(val)
    return ratio


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
        self.activations = feats.sign()  # .to(self.device)

    @torch.no_grad()
    def calc_swap(self, reg_factor=1.0):
        if self.activations is None:
            return 0
        # 转置后 unique(dim=0)
        self.activations = self.activations.t()  # => (features, N)
        unique_patterns = torch.unique(self.activations, dim=0).size(0)
        # print("unique_patterns * reg_factor:", unique_patterns, reg_factor)
        # / self.activations.size(0) * 100  # * reg_factor
        # return unique_patterns  # / self.activations.size(0) * 100
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


# ============ 9) 计算 SWAP 正则所需的 mu,sigma 的函数 ============

def calculate_mu_sigma(search_space, n_samples=1000):
    """
    SWAP正则化所需的 mu & sigma
    这里我们取中位数做 mu，标准差做 sigma
    """
    arr = []
    dummuy_input = torch.randn(1, 3, 224, 224)
    for _ in range(n_samples):
        op_codes = search_space.random_op_codes()
        width_codes = search_space.random_width_codes()
        model = search_space.get_model(op_codes, width_codes)
        model_info = get_model_complexity_info(model, dummuy_input)
        # param_mb = count_parameters_in_MB(model)
        param_mb = model_info['params'] / 1e6
        param_kb = param_mb * 1e3
        arr.append(param_kb)

    # 用 pandas 的 describe()
    series_ = pd.Series(arr)
    desc = series_.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    # 中位数
    mu = desc["mean"]
    # 标准差
    sigma = desc["std"]
    return mu, sigma
