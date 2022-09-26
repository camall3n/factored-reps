import torch

from factored_rl import configs

# ---------------------------------------------
# sparsity losses
# ---------------------------------------------

def compute_sparsity(data: torch.TensorType, cfg: configs.LossConfig):
    if cfg.name == 'sum_div_max':
        return sum_div_max_loss(data, epsilon=cfg.epsilon)
    elif cfg.name == 'unit_pnorm':
        return unit_pnorm_loss(data, p=cfg.p_norm)
    elif cfg.name == 'l2_div':
        return l2_div_loss(data, sigma=cfg.sigma)
    else:
        raise RuntimeError(f"Unknown sparsity loss: '{cfg.name}'")

def sum_div_max_loss(data, epsilon=1.0e-6):
    assert epsilon > 0
    magnitudes = torch.abs(data)
    l1 = torch.sum(magnitudes, dim=-1)
    lmax = torch.max(magnitudes, dim=-1)[0]
    residual_scores = l1 / (lmax + epsilon)
    return torch.mean(residual_scores)

def unit_pnorm_loss(data, p=2):
    assert p > 0
    magnitudes = torch.abs(data)
    l1 = torch.sum(magnitudes, dim=-1)
    normalized_data = magnitudes / l1
    residual_scores = -1 * torch.sum(normalized_data**p, dim=-1)
    return torch.mean(residual_scores)

def l2_div_loss(data, sigma=0.01):
    assert sigma > 0
    magnitudes = torch.abs(data)
    residual_scores = magnitudes / (magnitudes + sigma)
    return torch.mean(residual_scores)
