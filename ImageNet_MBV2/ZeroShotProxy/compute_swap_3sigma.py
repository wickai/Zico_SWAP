import numpy as np
import torch
import torch.nn as nn
from PlainNet.basic_blocks import RELU as PlainNetReLU


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def cal_regular_factor(model, mu, sigma):
    model_params = torch.as_tensor(count_parameters_in_MB(model)*1e3)
    print(f"Model parameters: {model_params} KB")
    print(f"Target mu: {mu} KB, sigma: {sigma} KB")
    
    # Ensure sigma is not too small to avoid numerical issues
    if sigma < 1e-6:
        sigma = 1.0
        print("WARNING: sigma too small, set to 1.0")
    
    # 计算参数量差异
    param_diff = torch.abs(model_params - mu)
    print(f"Parameter difference: {param_diff} KB")
    
    # 如果参数量差异太大，使用线性衰减而不是指数衰减
    if param_diff > 3 * sigma:
        # 使用线性衰减函数，确保最小值为0.1
        decay_factor = max(1.0 - (param_diff - 3*sigma) / (10*sigma), 0.1)
        print(f"Using linear decay factor: {decay_factor} (parameter difference too large)")
        return torch.tensor(decay_factor)
    
    # 正常情况下使用高斯衰减
    try:
        regular_factor = torch.exp(-(torch.pow((model_params-mu),2)/sigma))
        # 确保正则化因子不会太小
        if regular_factor < 0.1:
            regular_factor = torch.tensor(0.1)
            print(f"Regularization factor too small, setting to minimum: {regular_factor}")
        else:
            print(f"Calculated regularization factor: {regular_factor}")
        return regular_factor
    except Exception as e:
        print(f"Error in regularization calculation: {e}")
        return torch.tensor(1.0)  # Default to 1.0 if calculation fails

class SampleWiseActivationPatterns(object):
    def __init__(self, device):
        self.swap = -1 
        self.activations = None
        self.device = device

    @torch.no_grad()
    def collect_activations(self, activations):
        n_sample = activations.size()[0]
        n_neuron = activations.size()[1]
        print(f"Collecting activations with shape: {activations.shape}")

        if self.activations is None:
            self.activations = torch.zeros(n_sample, n_neuron).to(self.device)
            print(f"Initialized activations tensor with shape: {self.activations.shape}")

        # Check for NaN or Inf values
        if torch.isnan(activations).any() or torch.isinf(activations).any():
            print("WARNING: NaN or Inf values detected in activations")
            # Replace NaN/Inf with zeros
            activations = torch.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)

        self.activations = torch.sign(activations)
        print(f"Applied sign function, unique values: {torch.unique(self.activations)}")

    @torch.no_grad()
    def calSWAP(self, regular_factor):
        if self.activations is None:
            print("WARNING: activations is None in calSWAP")
            return 0.0
            
        print(f"Activation shape before transpose: {self.activations.shape}")
        self.activations = self.activations.T  # transpose the activation matrix: (samples, neurons) to (neurons, samples)
        print(f"Activation shape after transpose: {self.activations.shape}")
        
        unique_patterns = torch.unique(self.activations, dim=0)
        self.swap = unique_patterns.size(0)
        print(f"Number of unique activation patterns (SWAP): {self.swap}")
        print(f"Regularization factor: {regular_factor}")
        
        del self.activations
        self.activations = None
        torch.cuda.empty_cache()

        result = self.swap * regular_factor
        print(f"Final SWAP score: {result}")
        return result


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
        hook_count = 0
        for n, m in model.named_modules():
            # hook both nn.ReLU modules and PlainNet basic_blocks.RELU blocks
            if isinstance(m, nn.ReLU) or isinstance(m, PlainNetReLU):
                m.register_forward_hook(hook=self.hook_in_forward)
                hook_count += 1
                print(f"Registered hook for ReLU layer: {n}")
        print(f"Total hooks registered: {hook_count}")

    def hook_in_forward(self, module, input, output):
        if isinstance(input, tuple) and len(input[0].size()) == 4:
            self.interFeature.append(output.detach())
            print(f"Hook captured activation with shape: {output.shape}")

    def forward(self):
        self.interFeature = []
        with torch.no_grad():
            print(f"Input shape: {self.inputs.shape}")
            self.model.forward(self.inputs.to(self.device))
            print(f"Number of captured activation layers: {len(self.interFeature)}")
            if len(self.interFeature) == 0:
                print("WARNING: No activations captured. Check if model has ReLU layers.")
                return 0.0
                
            try:
                activtions = torch.cat([f.view(self.inputs.size(0), -1) for f in self.interFeature], 1)
                print(f"Concatenated activations shape: {activtions.shape}")
                self.swap.collect_activations(activtions)
                
                return self.swap.calSWAP(self.regular_factor)
            except Exception as e:
                print(f"Error in SWAP forward: {e}")
                import traceback
                traceback.print_exc()
                return 0.0

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
    print("\n==== Starting SWAP score computation ====\n")
    print(f"Regular: {regular}, GPU: {gpu}")
    print(f"Input shape: {inputs.shape}")
    
    # 检查模型是否有ReLU层
    relu_count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.ReLU) or isinstance(m, PlainNetReLU):
            relu_count += 1
    print(f"Number of ReLU layers in model: {relu_count}")
    
    if relu_count == 0:
        print("WARNING: No ReLU layers found in model. SWAP calculation will fail.")
        return 0.0
    
    # 如果需要正则化但没有提供mu和sigma，尝试从 model_list 计算
    if regular and (mu is None or sigma is None):
        if model_list is None:
            print("WARNING: regular=True but model_list is None. Cannot compute mu and sigma.")
            regular = False
        else:
            print(f"Computing mu and sigma from {len(model_list)} models")
            mu, sigma = compute_params_stats(model_list)
            print(f"Computed mu={mu}, sigma={sigma}")
    
    # 将模型放到指定GPU(如果gpu is not None)
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    inputs = inputs.to(device)
    
    # 确保sigma不为零，避免除零错误
    if regular and sigma is not None and sigma < 1e-6:
        print("WARNING: sigma is too small, setting to 1.0 to avoid division by zero")
        sigma = 1.0
    
    swap_evaluator = SWAP(model=model, inputs=inputs, device=device, regular=regular, mu=mu, sigma=sigma)
    swap_score = swap_evaluator.forward()
    
    print(f"Final SWAP score: {swap_score}")
    print("\n==== SWAP computation completed ====\n")
    
    return float(swap_score) if swap_score is not None else 0.0
