import torch
import numpy as np
import pandas as pd
import utils
import pathlib
import json
import solvers
import pinn
import argparse
import wandb
import visualize
import os
import logging

def get_args():
    parser = argparse.ArgumentParser(description='PINN launcher')
    parser.add_argument('--no_wandb', nargs='?', default=False, type=utils.str2bool, help="Prevent Wandb to log the run.")
    parser.add_argument('--save_model', nargs='?', default=False, type=utils.str2bool, help="Save the model.")
    parser.add_argument('--save_outputs', nargs='?', default=False, type=utils.str2bool, help="Save model results.")
    parser.add_argument('--visualize', default=False, type=utils.str2bool, help='Visualize the solution.')
    parser.add_argument('--config_file', default='configs.json', type=str, help='Input configuration file.')
    parser.add_argument('--num_seeds', default=0, type=int, help='Number of times each configuration should be tested.')
    
    args = parser.parse_args()
    return args

def pinn_train(cfgs, args, device):
    log_path = pathlib.Path(f"logs/{cfgs['system']}")
    log_path.mkdir(exist_ok=True, parents=True)

    """
    Parse arguments
    cfgs = {"id": 0,
            "system": "convection",
            "seed": 0,
            "N_f": 1000,
            "optimizer_name": "LBFGS",
            "lr": 0.1,
            "L": 1,
            "epochs": 0,
            "layers": "[50, 50, 50, 50, 1]",
            "net": "DNN",
            "u0_str": "sin(x)",
            "xgrid": 256,
            "tgrid": 100,
            "parameters": {
                "nu": 0,
                "rho": 0,
                "beta": 10
            },
            "curr_on": "beta",
            "curr_step": 1,
            "curr_initval": 0,
            "adaptive_step": 0.1,
            "seq2seq_step": 0.1,
            "training": "adaptive",
            "activation": "tanh"}
    """
    if cfgs['training'] != 'curriculum':
        for k in ['curr_on', 'curr_step', 'curr_initval']:
            cfgs[k] = None
    
    if cfgs['training'] != 'seq2seq':
        cfgs['seq2seq_step'] = None
    
    if cfgs['training'] != 'adaptive':
        cfgs['adaptive_step'] = None            

    if cfgs['layers'] is not None and isinstance(cfgs['layers'], str):
        cfgs['layers'] = eval(cfgs['layers'])

    curr_params = {
        'curr_on': cfgs['curr_on'], 
        'curr_step': cfgs['curr_step'],
        'curr_initval': cfgs['curr_initval']
    }
    
    # Define system-specific parameter settings
    system_parameters = {
        'convection': {'nu': 0., 'rho': 0.},
        'reaction': {'nu': 0., 'beta': 0.},
        'rd': {'beta': 0.}
    }

    # Apply system-specific settings
    if cfgs['system'] in system_parameters:
        for k, v in system_parameters[cfgs['system']].items():
            cfgs['parameters'][k] = v
    else:
        raise NotImplementedError('System not valid.')
    
    utils.set_seed(cfgs['seed'])
    
    """
    Calculate PDEs exact solutions
    """
    if cfgs['system'] == 'convection' or cfgs['system'] =='diffusion':
        u_vals = solvers.convection_diffusion(cfgs['u0_str'], cfgs['parameters']['nu'],
                                              cfgs['parameters']['beta'], 0, cfgs['xgrid'], 
                                              cfgs['tgrid'])
    elif cfgs['system'] == 'rd':
        u_vals = solvers.reaction_diffusion_discrete_solution(cfgs['u0_str'], cfgs['parameters']['nu'],
                                                              cfgs['parameters']['rho'], cfgs['xgrid'],
                                                              cfgs['tgrid'])
    elif cfgs['system'] == 'reaction':
        u_vals = solvers.reaction_solution(cfgs['u0_str'], cfgs['parameters']['rho'],
                                           cfgs['xgrid'], cfgs['tgrid'])
    else:
        raise NotImplementedError('System not valid.')

    """
    Process data
    """
    x = np.linspace(0, 2*np.pi, cfgs['xgrid'], endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, cfgs['tgrid']).reshape(-1, 1)

    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    Dom = np.hstack((X.reshape(-1, 1), T.reshape(-1, 1))) # all the (x, t) "test" data

    U_vals = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
    U_field = U_vals.reshape(t.size, x.size) # Exact solution on the (x, t) grid

    # Collocation points (exclude IC and BC)
    x_inner = x[1:]
    t_inner = t[1:]
    X_inner, T_inner = np.meshgrid(x_inner, t_inner)
    Dom_inner = np.hstack((X_inner.reshape(-1, 1), T_inner.reshape(-1, 1)))
    Dom_train = utils.sample_random(Dom_inner, cfgs['N_f']) # training dataset

    # Initial conditions
    Dom_init = np.hstack((X[0, :].reshape(-1, 1), T[0, :].reshape(-1, 1))) # ([-x_end, +x_end], 0)
    U_init = U_field[0:1, :].T # u([-x_end, +x_end], 0)

    # Lower boundary conditions
    Dom_lbound = np.hstack((X[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1))) # (x_min, [t])
    U_lbound = U_field[:, 0:1]

    # Upper boundary conditions: at x=2*pi
    Dom_ubound = np.hstack((np.array([2 * np.pi] * t.shape[0]).reshape(-1, 1), t))
    U_ubound = U_field[:, -1].reshape(-1, 1)

    if cfgs['layers'][0] != Dom_train.shape[-1]:
        cfgs['layers'].insert(0, Dom_train.shape[-1])

    """
    Train model
    """
    if not args.no_wandb:
        tags = [cfgs['training'], cfgs['system'], str(cfgs['seed'])]
        wandb.init(config=cfgs, project="pinn", tags=tags)
    
    print(120*'-')
    print(f"ID: {cfgs['id']} SEED: {cfgs['seed']}")
    model = pinn.PINN_pbc(cfgs['system'], Dom_init, U_init, Dom_train, Dom_lbound, Dom_ubound,
                        cfgs['layers'], cfgs['parameters'], cfgs['optimizer_name'], cfgs['training'],
                        cfgs['N_f'], cfgs['lr'], cfgs['net'], cfgs['L'], cfgs['activation'], device)
    
    
    model.train(cfgs['epochs'], curriculum=curr_params, seq2seq_step=cfgs['seq2seq_step'], adaptive_step=cfgs['adaptive_step'])
    eval_dict = model.validate(Dom, U_real=U_field, is_last=True)
    
    if wandb.run is not None:
        if cfgs['training'] == 'curriculum':
            wandb.log({model.curr_on: getattr(model, model.curr_on)}, step=model.iter)
    
        wandb.finish()

    outputs_file = f"{cfgs['system']}_{cfgs['training']}_id{cfgs['id']}_nu{cfgs['parameters']['nu']}_beta{cfgs['parameters']['beta']}_rho{cfgs['parameters']['rho']}"
    model_file = f"{outputs_file}_seed{cfgs['seed']}"

    if args.save_model:
        path = log_path / 'models'
        path.mkdir(parents=True, exist_ok=True)

        file_path = pathlib.Path(f'{path}/') / model_file
        torch.save(model, f"{file_path}.pt")

    if args.save_outputs:
        path = log_path / 'results'
        path.mkdir(parents=True, exist_ok=True)
        file_path = pathlib.Path(f'{path}/') / outputs_file

        # Flatten each dictionary in the list
        # Convert the flattened list of dictionaries to a DataFrame
        flattened_cfgs = utils.flatten_dict(cfgs)
        result_dict = {**flattened_cfgs, **eval_dict}

        df = pd.DataFrame([result_dict])
        if os.path.isfile(f"{file_path}.csv"):
            df.to_csv(f"{file_path}.csv", mode='a', index=False, header=False)
        else:
            df.to_csv(f"{file_path}.csv", mode='w', index=False, header=True)

    if args.visualize:
        path = log_path / 'plots'
        path.mkdir(parents=True, exist_ok=True)
        
        U_pred = model.predict(Dom).reshape(t.size, x.size)
        visualize.u_exact(U_field, x, t, cfgs, pathlib.Path(path) / f"exact_{model_file}.pdf")
        visualize.u_predict(U_field, U_pred, x, t, cfgs, pathlib.Path(path) / f"upredicted_{model_file}.pdf")
        visualize.u_diff(U_field, U_pred, x, t, cfgs, pathlib.Path(path) / f"udiff_{model_file}.pdf")
        
def main(args, device):
    config_filepath = os.path.join('cfg', args.config_file)

    if not os.path.isfile(config_filepath):
        logging.error(f"Configuration file '{config_filepath}' does not exist.")
        return
    
    try:
        with open(config_filepath, 'r') as file:
            configs = json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Error reading JSON configuration file: {e}")
        return
        
    for config in configs:
        base_seed = config['seed']
        num_seeds = args.num_seeds
        seeds_list = [2, 3, 5, 8, 14, 19, 21, 60, 89, 105]
        
        if num_seeds == 0: # num_seeds is the number of times each configuration should be tested
            for seed in seeds_list:
                config['seed'] = seed
                pinn_train(config, args, device)
        else:
            for i in range(num_seeds):
                current_seed = base_seed + i
                config['seed'] = current_seed
                pinn_train(config, args, device) 


if __name__ == '__main__':
    device = torch.device(f'cuda:{utils.free_cuda_id()}') if torch.cuda.is_available() else 'cpu'

    args = get_args()
    main(args, device)
        
