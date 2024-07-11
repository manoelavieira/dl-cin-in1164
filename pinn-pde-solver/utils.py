import numpy as np
import random
import torch
import argparse
import subprocess
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def flatten_dict(d, parent_key='', sep='_'):
    """ Flatten a nested dictionary """
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def sample_random(X_all, N):
    """ Given an array of (x,t) points, sample N points from this. """
    # set_seed(0) # this can be fixed for all N_f

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled

def function(u0: str):
    """ Initial condition, string --> function. """
    funs = {
        'sin(x)': lambda x: np.sin(x),
        '-sin(x)': lambda x: -np.sin(x),
        '-sin(pix)': lambda x: -np.sin(np.pi*x),
        '-sin(2pix)': lambda x: -np.sin(2*np.pi*x),
        'sin(pix)': lambda x: np.sin(np.pi*x),
        'sin^2(x)': lambda x: np.sin(x)**2,
        'sin(x)cos(x)': lambda x: np.sin(x)*np.cos(x),
        'x^2*cos(pix)': lambda x: x**2 * np.cos(np.pi*x),
        '0.1sin(x)': lambda x: 0.1*np.sin(x),
        '0.5sin(x)': lambda x: 0.5*np.sin(x),
        '10sin(x)': lambda x: 10*np.sin(x),
        '50sin(x)': lambda x: 50*np.sin(x),
        '1+sin(x)': lambda x: 1 + np.sin(x),
        '2+sin(x)': lambda x: 2 + np.sin(x),
        '6+sin(x)': lambda x: 6 + np.sin(x),
        '10+sin(x)': lambda x: 10 + np.sin(x),
        'sin(2x)': lambda x: np.sin(2*x),
        'tanh(x)': lambda x: np.tanh(x),
        '2x': lambda x: 2*x,
        'x^2': lambda x: x**2,
        'gauss':  lambda x: np.exp(-np.power(x - np.pi, 2.) / (2 * np.power(np.pi/4, 2.)))
    }
    return funs[u0]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def free_cuda_id():
    """ Get the list of GPUs via nvidia-smi """
    smi_query_result = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU", shell=True)

    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]

    return gpu_info.index(min(gpu_info))