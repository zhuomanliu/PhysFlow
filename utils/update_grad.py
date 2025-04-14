import torch
import numpy as np
import warp as wp
from mpm_solver_warp.mpm_utils import update_param

def update_grad_param(param, param_grad, n_particles, lrate=1.0, lower=-1.0, upper=-0.4, gn=False, scale=1., log_name=None, debug=False):
    grad = wp.to_torch(param_grad) / scale
    
    if gn:
        max_grad, min_grad = torch.max(grad), torch.min(grad)
        grad = (grad - min_grad) / (max_grad - min_grad) - 0.5 if max_grad - min_grad != 0 else torch.zeros_like(grad)

    if not debug:
        wp.launch(update_param, n_particles, [param, wp.from_torch(grad), lrate, upper, lower])

    if log_name is not None:
        print(f"- {log_name}: {torch.mean(wp.to_torch(param)).item()}, grad_{log_name}: {torch.mean(grad).item()}, lr_{log_name}: {lrate}")


def lr_scheduler(lr_init, lr_end, step, total_steps, warmup_steps=0, max_steps=None):
    if max_steps is None:
        max_steps = total_steps
    if step < warmup_steps:
        lr = float(step) / float(warmup_steps) * (lr_init - lr_end) + lr_end
    elif step < max_steps:
        lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos((step - warmup_steps) / (max_steps - warmup_steps) * np.pi))
    else:
        lr = lr_end
    return lr