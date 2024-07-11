import torch
import torch_optimizer as optim
import numpy as np

def choose_optimizer(optimizer_name: str, *params):
    if optimizer_name == 'LBFGS':
        return LBFGS(*params)
    elif optimizer_name == 'Adam':
        return Adam(*params)
    elif optimizer_name == 'SGD':
        return SGD(*params)
    else:
        raise NotImplementedError('Optimizer not valid.')

def LBFGS(model_param, lr=1.0, max_iter=100000, max_eval=None, history_size=50,
          tolerance_grad=1e-7, tolerance_change=1e-7, line_search_fn="strong_wolfe"):
    
    optimizer = torch.optim.LBFGS(
        model_param,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn)

    return optimizer

def Adam(model_param, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    optimizer = torch.optim.Adam(model_param, lr=lr)

    return optimizer

def SGD(model_param, lr=1e-4, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
    optimizer = torch.optim.SGD(model_param, lr=lr, momentum=momentum, dampening=dampening,
                                weight_decay=weight_decay, nesterov=False)

    return optimizer