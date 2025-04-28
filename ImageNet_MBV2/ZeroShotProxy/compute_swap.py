import numpy as np
import torch
import torch.nn as nn


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def cal_regular_factor(model, mu, sigma):
    model_params = torch.as_tensor(count_parameters_in_MB(model)*1e3)
    regular_factor =  torch.exp(-(torch.pow((model_params-mu),2)/sigma))
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
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(hook=self.hook_in_forward)

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach()) 

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            self.model.forward(self.inputs.to(self.device))
            if len(self.interFeature) == 0: return
            activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)         
            self.swap.collect_activations(activtions)
            
            return self.swap.calSWAP(self.regular_factor)

    def _hook_fn(self, module, inp, out):
        """Forward hook函数，用于收集ReLU层的输出特征
        
        Args:
            module: PyTorch模块
            inp: 输入张量
            out: 输出张量
        """
        # out: shape (N, C, H, W) --> reshape to (N, -1)
        feats = out.detach().reshape(out.size(0), -1)
        self.inter_feats.append(feats)

    def _clear_hooks(self, hooks):
        for h in hooks:
            h.remove()
        hooks.clear()


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

def compute_swap_score(gpu, model, inputs, regular=False, mu=None, sigma=None, model_list=None):
    """计算SWAP分数
    
    Args:
        gpu: GPU设备号，如果使用CPU则为None
        model: 要计算的模型
        inputs: 输入数据
        regular: 是否使用参数量正则化
        mu: 目标参数量均值（KB），如果为None且regular=True则从 model_list 计算
        sigma: 参数量标准差（KB），如果为None且regular=True则从 model_list 计算
        model_list: 用于计算mu和sigma的模型列表
    
    Returns:
        float: SWAP分数
    """
    # 如果需要正则化但没有提供mu和sigma，尝试从 model_list 计算
    if regular and (mu is None or sigma is None):
        if model_list is None:
            raise ValueError("When regular=True and mu/sigma not provided, model_list must be provided")
        mu, sigma = compute_params_stats(model_list)
    
    # 将模型放到指定GPU(如果gpu is not None)
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    inputs = inputs.to(device)
    
    swap_evaluator = SWAP(model=model, inputs=inputs, device=device, regular=regular, mu=mu, sigma=sigma)
    swap_score = swap_evaluator.forward()
    return float(swap_score) if swap_score is not None else 0.0
