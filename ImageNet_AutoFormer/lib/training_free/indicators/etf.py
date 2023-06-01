import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

# def kaiming_normal_fanin_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         if m.affine:
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

# def kaiming_normal_fanout_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#         if m.affine:
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

# def init_model(model, method='kaiming_norm_fanin'):
#     if method == 'kaiming_norm_fanin':
#         model.apply(kaiming_normal_fanin_init)
#     elif method == 'kaiming_norm_fanout':
#         model.apply(kaiming_normal_fanout_init)
#     else:
#         raise NotImplementedError
#     return model


def compute_nas_score(model, device, trainloader, resolution, batch_size):
    model.train()
    info = {}

    # init_model(model, 'kaiming_norm_fanin')

    if trainloader == None:
        input_ = torch.randn(size=[batch_size, 3, resolution, resolution], device=device)
    else:
        input_ = next(iter(trainloader))[0]
        input_ = input_.to(device)
    
    blk_features = model.extract_blk_features(input_)

    ################ fwrd pca score ################
    """
    pca score across residual block features / normalize each score by upper bound
    """
    info_flow_scores = []
    expressivity_scores = []
    prev_feat = None
    for i in range(len(blk_features)):
        feat = blk_features[i].detach().clone()
        ### avoid duplicated features
        if prev_feat is None:
            prev_feat = blk_features[i].detach().clone()
        else:
            assert not torch.all(feat == prev_feat)
            prev_feat = blk_features[i].detach().clone()
        ### 
        b,n,c = feat.size()
        feat = feat.view(b*n,c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.transpose(1,0),feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma) # faster version for computing eignevalues, can be adopted since sigma is symmetric
        prob_s = s / s.sum()
        score = (-prob_s)*torch.log(prob_s+1e-8)
        score = score.sum().item()
        info_flow_scores.append(score)
        expressivity_scores.append(score / np.log(c)) # normalize by an upper bound (= np.log(c))
    info_flow_scores = np.array(info_flow_scores)
    info_flow = np.min(info_flow_scores[1:] - info_flow_scores[:-1])
    expressivity = np.mean(expressivity_scores)
    #################################################

    ################ spec norm score ##############
    """
    spec norm score across residual block features
    """
    scores = []
    for i in reversed(range(1, len(blk_features))):
        f_out = blk_features[i]
        f_in = blk_features[i-1]
        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size()==g_in.size() and torch.all(g_in == g_out):
            continue
        else:
            if g_out.size(1) != g_in.size(1) or g_out.size(1) != g_in.size(1):
                raise NotImplementedError
            bo,no,co = g_out.size()
            bi,ni,ci = g_in.size()
            g_out = g_out.view(bo*no,co)
            g_in = g_in.view(bi*ni,ci)
            mat = torch.mm(g_in.transpose(1,0),g_out) / (bo*no)
            ### make faster on cpu
            if mat.size(0) < mat.size(1):
                mat = mat.transpose(0,1)
            ###
            s = torch.linalg.svdvals(mat)
            scores.append(-s.max().item() - 1/(s.max().item()+1e-6)+2)
    bkwd_norm_score = np.mean(scores)
    #################################################

    info['expressivity'] = float(expressivity) if not np.isnan(expressivity) else -np.inf
    info['info_flow'] = float(info_flow) if not np.isnan(info_flow) else -np.inf
    info['trainability'] = float(bkwd_norm_score) if not np.isnan(bkwd_norm_score) else -np.inf
    info['complexity'] = float(model.get_complexity(model.patch_embed_super.num_patches+1))

    return info