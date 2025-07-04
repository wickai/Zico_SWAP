import numpy as np
import torch
import torch.nn as nn
from PlainNet.basic_blocks import RELU as PlainNetReLU

# 全局变量用于存储预先计算的mu和sigma
_GLOBAL_MU = None
_GLOBAL_SIGMA = None
_GLOBAL_STATS_COMPUTED = False  # 标记是否已经计算过统计信息


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def cal_regular_factor(model, mu, sigma):
    model_params = torch.as_tensor(count_parameters_in_MB(model)*1e3)
    regular_factor = torch.exp(-(torch.pow((model_params-mu),2)/sigma))
    return regular_factor

class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1 
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)  

        self.activations = torch.sign(activations)

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        self.activations = self.activations.T  # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        self.swap = torch.unique(self.activations, dim=0).size(0)
        
        del self.activations
        self.activations = None
        torch.cuda.empty_cache()

        return self.swap * regular_factor


class SWAP:
    def __init__(self, model=None, inputs=None, device='cpu', seed=0, regular=True, mu=None, sigma=None):
        self.model = model
        self.interFeature = []
        self.seed = seed
        self.regular = regular
        self.regular_factor = 1
        self.mu = mu
        self.sigma = sigma
        self.inputs = inputs
        self.device = device
        self.reinit(self.model, self.seed)

    def reinit(self, model=None, seed=None):
        if model is not None:
            self.model = model
            self.register_hook(self.model)
            self.swap = SampleWiseActivationPatterns(self.device)
            if self.regular and self.mu is not None and self.sigma is not None:
                self.regular_factor = cal_regular_factor(self.model, self.mu, self.sigma).item()
 
        if seed is not None and seed != self.seed:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def clear(self):
        self.swap = SampleWiseActivationPatterns(self.device)
        del self.interFeature
        self.interFeature = []
        torch.cuda.empty_cache()

    def register_hook(self, model):
        for n, m in model.named_modules():
            if isinstance(m, nn.ReLU) or isinstance(m, PlainNetReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0: 
                return 0.0
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)         
            self.swap.collect_activations(activtions)
            
            return self.swap.calSWAP(self.regular_factor)


def compute_params_stats(model_list, top_k=100):
    """计算模型参数量的统计信息
    
    Args:
        model_list: 模型列表
        top_k: 使用前k个模型来计算统计信息
    
    Returns:
        tuple: (mu, sigma) mu为参数量均值（KB），sigma为标准差（KB）
    """
    if not model_list:
        raise ValueError("model_list cannot be empty")
    
    # 取前k个模型
    models = model_list[:min(top_k, len(model_list))]
    
    # 计算每个模型的参数量（KB）
    param_sizes = [count_parameters_in_MB(model) * 1024 for model in models]  # 转换为KB
    
    # 计算均值和标准差
    mu = sum(param_sizes) / len(param_sizes)
    sigma = (sum((x - mu) ** 2 for x in param_sizes) / len(param_sizes)) ** 0.5
    
    return mu, sigma


def precompute_params_stats(model_list, top_k=1000):
    """预先计算模型参数量的统计信息并存储在全局变量中
    
    Args:
        model_list: 模型列表
        top_k: 使用前k个模型来计算统计信息，默认使用1000个模型
    
    Returns:
        tuple: (mu, sigma) mu为参数量均值（KB），sigma为标准差（KB）
    """
    global _GLOBAL_MU, _GLOBAL_SIGMA, _GLOBAL_STATS_COMPUTED
    
    # 如果已经计算过，直接返回
    if _GLOBAL_STATS_COMPUTED and _GLOBAL_MU is not None and _GLOBAL_SIGMA is not None:
        print(f"使用已预先计算的统计信息 - mu: {_GLOBAL_MU} KB, sigma: {_GLOBAL_SIGMA} KB")
        return _GLOBAL_MU, _GLOBAL_SIGMA
    
    # 计算并存储
    _GLOBAL_MU, _GLOBAL_SIGMA = compute_params_stats(model_list, top_k)
    _GLOBAL_STATS_COMPUTED = True  # 标记为已计算
    print(f"预先计算的统计信息 - mu: {_GLOBAL_MU} KB, sigma: {_GLOBAL_SIGMA} KB (基于 {min(top_k, len(model_list))} 个模型)")
    
    return _GLOBAL_MU, _GLOBAL_SIGMA

def compute_swap_score(gpu, model, inputs, regular=False, mu=None, sigma=None, model_list=None, use_global_stats=True):
    """计算SWAP分数
    
    Args:
        gpu: GPU设备号，如果使用CPU则为None
        model: 要计算的模型
        inputs: 输入数据
        regular: 是否使用参数量正则化
        mu: 目标参数量均值（KB），如果为None且regular=True则从 model_list 计算或使用全局值
        sigma: 参数量标准差（KB），如果为None且regular=True则从 model_list 计算或使用全局值
        model_list: 用于计算mu和sigma的模型列表
        use_global_stats: 是否使用全局预先计算的统计信息，默认为True
    
    Returns:
        float: SWAP分数
    """
    global _GLOBAL_MU, _GLOBAL_SIGMA, _GLOBAL_STATS_COMPUTED
    
    # 检查模型是否有ReLU层
    relu_count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.ReLU) or isinstance(m, PlainNetReLU):
            relu_count += 1
    
    if relu_count == 0:
        return 0.0
    
    # 如果需要正则化但没有提供mu和sigma
    if regular and (mu is None or sigma is None):
        # 优先使用全局预先计算的统计信息
        if use_global_stats and _GLOBAL_STATS_COMPUTED and _GLOBAL_MU is not None and _GLOBAL_SIGMA is not None:
            mu = _GLOBAL_MU
            sigma = _GLOBAL_SIGMA
            print(f"使用全局预先计算的统计信息 - mu: {mu} KB, sigma: {sigma} KB")
        # 如果没有全局统计信息且提供了model_list，则计算
        elif model_list is not None:
            mu, sigma = compute_params_stats(model_list)
            print(f"使用当前种群计算的统计信息 - mu: {mu} KB, sigma: {sigma} KB (基于 {len(model_list)} 个模型)")
        # 如果都没有，则禁用正则化
        else:
            print("没有可用的统计信息，禁用正则化")
            regular = False
    
    # 将模型放到指定GPU(如果gpu is not None)
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    inputs = inputs.to(device)
    
    # 确保sigma不为零，避免除零错误
    if regular and sigma is not None and sigma < 1e-6:
        sigma = 1.0
    
    swap_evaluator = SWAP(model=model, inputs=inputs, device=device, regular=regular, mu=mu, sigma=sigma)
    swap_score = swap_evaluator.forward()
    
    return float(swap_score) if swap_score is not None else 0.0
