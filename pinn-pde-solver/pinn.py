import torch
import dnn
import numpy as np
import optimizers
import utils
import wandb

import plotly.express as px
import plotly.io as pio

import warnings
warnings.filterwarnings('ignore')

class PINN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, Dom_init, U_init, Dom_train, Dom_lbound, Dom_ubound, layers,
                 parameters, optimizer_name, training, N_f, lr=1e-3, net='DNN', L=1, activation='tanh',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.system = system
        self.net = net
        
        # CUDA support
        print(f"Using device: {self.device}")
        
        self.x = Dom_init[:, 0]
        self.t = Dom_lbound[:, 1]

        # Initial condititon (t=0)
        self.x_init = torch.tensor(Dom_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(Dom_init[:, 1:2], requires_grad=True).float().to(device)
        
        self.u = torch.tensor(U_init, requires_grad=True).float().to(device)
        
        # Collocation points
        self.x_f = torch.tensor(Dom_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(Dom_train[:, 1:2], requires_grad=True).float().to(device)
        
        # LB/BC (x=0 and x=~6.28)
        self.x_lbound = torch.tensor(Dom_lbound[:, 0:1], requires_grad=True).float().to(device)
        self.t_lbound = torch.tensor(Dom_lbound[:, 1:2], requires_grad=True).float().to(device)
        self.x_ubound = torch.tensor(Dom_ubound[:, 0:1], requires_grad=True).float().to(device)
        self.t_ubound = torch.tensor(Dom_ubound[:, 1:2], requires_grad=True).float().to(device)
        
        G = np.full(Dom_train.shape[0], 0.) # Source for convection: till now it is hardcoded
        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        # Apply parameters
        # parameters = {nu: <>, beta: <>, ....}
        for k, v in parameters.items():
            setattr(self, k, v)

        if self.net == 'DNN':
            self.dnn = dnn.DNN(layers, activation)
        else: # "pretrained" can be included in model path
            self.dnn = torch.load(net).dnn
        self.dnn.to(device)

        self.layers = layers
        self.L = L
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = optimizers.choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)
        self.iter = 0
        self.curr_on = None
        self.training = training
        self.N_f = N_f

    def net_u(self, x, t):
        """
        The standard DNN that takes (t) | (x,t) --> u.
        If X has more dimension, it must be a list of column tensors (e.g. [x, t])
        """
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ 
        Neural Network + Autograd + Physics information
        Autograd for calculating the residual for different systems.
        """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Get residuals
        if 'convection' in self.system:
            f = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2
        else:
            raise NotImplementedError('System not valid.')
        
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_lbound, x_ubound):
        """For taking BC derivatives."""
        u_lb_x = torch.autograd.grad(
            u_lb, x_lbound,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
        )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_ubound,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
        )[0]

        return u_lb_x, u_ub_x

    def boundary_loss(self, u_pred_lbound, u_pred_ubound):
        """Boundary loss for PBC"""
        loss_b = torch.mean((u_pred_lbound - u_pred_ubound) ** 2)
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lbound, u_pred_ubound, self.x_lbound, self.x_ubound)
            loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)

        return loss_b

    def calc_grad_norm(self):
        grad_norm = 0
        
        for p in self.dnn.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5

        return grad_norm

    def loss_pinn(self):
        """ Loss function. """
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
 
        # Prediction for initial conditions (IC)
        u_pred = self.net_u(self.x_init, self.t_init)
        loss_u = torch.mean((self.u - u_pred) ** 2)

        # Prediction for Lower and Upper BC
        u_pred_lbound = self.net_u(self.x_lbound, self.t_lbound)
        u_pred_ubound = self.net_u(self.x_ubound, self.t_ubound)
        loss_b = self.boundary_loss(u_pred_lbound, u_pred_ubound)
                    
        # Prediction for collocation points
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_pred ** 2)

        # Combine loss terms
        loss = loss_u + loss_b + (self.L * loss_f)

        if loss.requires_grad:
            loss.backward()

        grad_norm = self.calc_grad_norm()
        if self.iter % 100 == 0:
            print('epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_b.item(), loss_f.item()))

        if wandb.run is not None:
            wandb.log({"bound_loss": loss_b}, step=self.iter)
            wandb.log({"fun_loss": loss_f}, step=self.iter)
            wandb.log({"init_loss": loss_u}, step=self.iter)
            wandb.log({"loss": loss}, step=self.iter)
        
        self.iter += 1
        if wandb.run is not None:
            wandb.log({'epoch': self.iter})

        return loss

    def predict(self, X):
        x = torch.tensor(X[:, 0:1]).float().to(self.device)
        t = torch.tensor(X[:, 1:2]).float().to(self.device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()

        return u

    def validate(self, Dom, U_real, is_last=False):
        U_pred = self.predict(Dom).reshape(self.t.size, self.x.size)
        
        mae = np.mean(np.abs(U_real - U_pred))

        if self.training != 'vanilla':
            if wandb.run is not None:
                wandb.log({'valid_mae': mae}, step=self.iter)

        eval_dict = {}
        if is_last:
            error_u_abs = mae
            error_u_relative = np.linalg.norm(U_real - U_pred, 2) / np.linalg.norm(U_real, 2)            

            if wandb.run is not None:
                fig = px.imshow(U_pred.T, x=self.t, y=self.x, aspect='auto', origin='lower', 
                                labels={'x': 'T', 'y': 'X'},
                                color_continuous_scale=px.colors.sequential.Turbo)

                tab_data = wandb.Table(
                    columns=['Relative error', 'Absolute error'],
                    data=[[error_u_relative, error_u_abs]]
                )

                wandb.log({'Evaluation': fig})
                wandb.log({'Model evaluation': tab_data})
            
            eval_dict = {
                "total_epochs": self.iter,   
                "error_u_abs": error_u_abs,
                "error_u_relative": error_u_relative,
            }

            print('Error rel: %e' % (eval_dict['error_u_relative']))
            print('Error abs: %e' % (eval_dict['error_u_abs']))
            print('Total epochs: %d\n' % (eval_dict['total_epochs']))
        
        return eval_dict

    def config_curriculum_learning(self, curriculum):
        """
        Configure curriculum learning by setting the range of values for the specified parameter.

        Parameters:
            curriculum: dictionary with curriculum regularization setup (curr_on, curr_initval, curr_step)
        """
        self.curr_on = curriculum['curr_on']
        param_stop = getattr(self, self.curr_on)
        
        print(f"Using curriculum learning on param {curriculum['curr_on']} with target {param_stop}")
        self.curr_range = np.arange(curriculum['curr_initval'], param_stop + curriculum['curr_step'], curriculum['curr_step'])
        setattr(self, self.curr_on, self.curr_range[0])
    
    def config_seq2seq_learning(self, seq2seq_step):
        """
        Configure seq2seq learning by setting up the mesh grid and initializing parameters.

        Parameters:
            seq2seq_step: The step size for seq2seq learning.
        """

        # Setup inner grid points
        x_inner = self.x[1:]
        t_inner = self.t[1:]
        X_inner, T_inner = np.meshgrid(x_inner, t_inner)
        X, T = np.meshgrid(self.x, self.t) # all grid points

        # Initialize domain boundaries and collocation points
        self.Dom_lbound = np.hstack((X[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1))) # all LB points
        self.Dom_ubound = np.hstack((np.array([2 * np.pi] * self.t.size).reshape(-1, 1), self.t.reshape(-1, 1))) # all UB points
        self.Dom_inner = np.hstack((X_inner.reshape(-1, 1), T_inner.reshape(-1, 1))) # all collocation points
        
        # Number of collocation points to sample each time step
        self.N_f = int(self.N_f * seq2seq_step) 

        # Initialize source term tensor for convection
        G = np.full(self.N_f, 0.)
        self.G = torch.tensor(G, requires_grad=True).float().to(self.device)
        self.G = self.G.reshape(-1, 1)
        
        # Define the seq2seq range
        self.seq2seq_range = np.arange(0, 1 + seq2seq_step, seq2seq_step)
        print(f"Using seq2seq learning with time increment {seq2seq_step}")
    
    def config_adaptive_learning(self, adaptive_step, version=1):
        """
        Configure adaptive learning by setting up the mesh grid and initializing parameters.

        Parameters:
            adaptive_step: The step size for adaptive learning.
        """

        # Setup inner grid points
        x_inner = self.x[1:]
        t_inner = self.t[1:]
        X_inner, T_inner = np.meshgrid(x_inner, t_inner)
        X, T = np.meshgrid(self.x, self.t) # all grid points

        # Initialize domain boundaries and collocation points
        self.Dom_lbound = np.hstack((X[:, 0].reshape(-1, 1), T[:, 0].reshape(-1, 1)))
        self.Dom_ubound = np.hstack((np.array([2 * np.pi] * self.t.size).reshape(-1, 1), self.t.reshape(-1, 1)))
        self.Dom_inner = np.hstack((X_inner.reshape(-1, 1), T_inner.reshape(-1, 1)))
        
        # Initialize training domain and number of collocation points
        self.N_f_tot = self.N_f
        self.N_f = 0
        self.Dom_train = np.empty((0, 2))

        # Define the adaptive learning range
        if (version == 1):
            self.adaptive_interval_lst = np.arange(0, 1 + adaptive_step, adaptive_step)
            self.adaptive_npoints_lst = [int(interval * self.N_f_tot) for interval in self.adaptive_interval_lst[1:]]
            print(f"Using adaptive learning v1 with time increment {adaptive_step}")
            print(f"Using adaptive learning v1 with number of points {self.adaptive_npoints_lst}")
        elif (version == 2):
            n_points = int((1 / adaptive_step) + 1)
            b = 0.3

            intervals = [np.exp(-b * x) for x in range(n_points)]
            total_interval = sum(intervals)
            normalized_intervals = [interval / total_interval for interval in intervals]

            # Generate the list with the specified intervals
            self.adaptive_interval_lst = [0]
            for interval in normalized_intervals:
                self.adaptive_interval_lst.append(self.adaptive_interval_lst[-1] + interval)

            self.adaptive_interval_lst[-1] = 1
            self.adaptive_npoints_lst = [int(interval * self.N_f_tot) for interval in self.adaptive_interval_lst[1:]]
            print(f"Using adaptive learning v2 with number of points {self.adaptive_npoints_lst}")
        else:
            raise NotImplementedError('Version not valid.')
    
    def update_seq2seq_learning(self, curr_val, next_val):
        """
        Update seq2seq learning by sampling the training domain within the current time interval.

        Parameters:
            curr_val: The current time value in the seq2seq range.
            next_val: The next time value in the seq2seq range.
        """

        # Select the collocation points within the current step range
        in_range_idx = (self.Dom_inner[:, 1] >= curr_val) & (self.Dom_inner[:, 1] <= next_val)
        Dom_train = utils.sample_random(self.Dom_inner[in_range_idx], self.N_f)

        # Select the boundary points within the current step range
        lbound_idx = (self.Dom_lbound[:, 1] >= curr_val) & (self.Dom_lbound[:, 1] < next_val)
        ubound_idx = (self.Dom_ubound[:, 1] >= curr_val) & (self.Dom_ubound[:, 1] < next_val)
        Dom_lbound = self.Dom_lbound[lbound_idx]
        Dom_ubound = self.Dom_lbound[ubound_idx] 

        # Convert sampled data to tensors and set training configuration for the current step
        self.x_f = torch.tensor(Dom_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(Dom_train[:, 1:2], requires_grad=True).float().to(self.device)
        self.x_lbound = torch.tensor(Dom_lbound[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_lbound = torch.tensor(Dom_lbound[:, 1:2], requires_grad=True).float().to(self.device)
        self.x_ubound = torch.tensor(Dom_ubound[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_ubound = torch.tensor(Dom_ubound[:, 1:2], requires_grad=True).float().to(self.device)

    def update_adaptive_learning(self, adaptive_step, curr_val, next_val, index):
        """
        Update adaptive learning by sampling the training domain and incrementing the number of collocation points.
        
        Parameters:
            adaptive_step: The step size for adaptive learning.
            curr_val: The current time value in the adaptive learning range.
            next_val: The next time value in the adaptive learning range.
        """
        
        # Sample training domain points within the current time interval [0, time next_val]
        self.N_f = self.adaptive_npoints_lst[index]
        in_range_idx = (self.Dom_inner[:, 1] <= next_val)
        self.Dom_train = utils.sample_random(self.Dom_inner[in_range_idx], self.N_f)
        print(f"Number of collocation points: {self.N_f}")

        # Filter boundary domain points in range [0, time next_val]
        Dom_lbound = self.Dom_lbound[(self.Dom_lbound[:, 1] >= 0) & (self.Dom_lbound[:, 1] < next_val)]
        Dom_ubound = self.Dom_ubound[(self.Dom_ubound[:, 1] >= 0) & (self.Dom_ubound[:, 1] < next_val)]
    
        # Update the source term tensor for convection (G)
        G = np.full(self.N_f, 0.)
        self.G = torch.tensor(G, requires_grad=True).float().to(self.device)
        self.G = self.G.reshape(-1, 1)

        # Convert training and boundary points to tensors
        self.x_f = torch.tensor(self.Dom_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(self.Dom_train[:, 1:2], requires_grad=True).float().to(self.device)
        self.x_lbound = torch.tensor(Dom_lbound[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_lbound = torch.tensor(Dom_lbound[:, 1:2], requires_grad=True).float().to(self.device)
        self.x_ubound = torch.tensor(Dom_ubound[:, 0:1], requires_grad=True).float().to(self.device)
        self.t_ubound = torch.tensor(Dom_ubound[:, 1:2], requires_grad=True).float().to(self.device)

    def train(self, epochs=3000, curriculum:dict=None, seq2seq_step=0, adaptive_step=0, Dom=None, U_real=None):
        """
        Train the model using specified training methods.
        """
        self.dnn.train()

        if self.training == 'vanilla':
            self.optimizer.step(self.loss_pinn)
        elif self.training == 'curriculum':
            self._train_curriculum(epochs, curriculum, Dom, U_real)
        elif self.training == 'seq2seq':
            self._train_seq2seq(epochs, seq2seq_step, Dom, U_real)
        elif self.training == 'adaptive':
            self._train_adaptive(epochs, adaptive_step, Dom, U_real)
        else:
            raise NotImplementedError('Training method not valid.')

    def _train_curriculum(self, epochs, curriculum, Dom, U_real):
        self.config_curriculum_learning(curriculum)
        print(f'Current curriculum range: {self.curr_range}')
        print(120*'-')
        
        for curr_val in self.curr_range:
            setattr(self, self.curr_on, curr_val) # update parameter value
            
            print(f'{self.curr_on}: {getattr(self, self.curr_on)}')
            self.optimizer.step(self.loss_pinn)
            
            if Dom is not None:
                self.dnn.eval()
                self.validate(Dom, U_real)
    
    def _train_seq2seq(self, epochs, seq2seq_step, Dom, U_real):
        self.config_seq2seq_learning(seq2seq_step)
        print(f'Current seq2seq range: {self.seq2seq_range}')
        print(120*'-')
        
        for curr_val, next_val in zip(self.seq2seq_range[:-1], self.seq2seq_range[1:]):
            print(f'Time interval: {curr_val} to {next_val}')

            self.update_seq2seq_learning(curr_val, next_val)
            self.optimizer.step(self.loss_pinn)

    def _train_adaptive(self, epochs, adaptive_step, Dom, U_real):
        self.config_adaptive_learning(adaptive_step, version=1) # adaptive learning has 2 versions: v1 and v2 
        print(f'Current adaptive interval list: {self.adaptive_interval_lst}')
        print(120*'-')
        
        for index, (curr_val, next_val) in enumerate(zip(self.adaptive_interval_lst[:-1], self.adaptive_interval_lst[1:])):
            print(f'Time interval: {0} to {next_val}')

            self.update_adaptive_learning(adaptive_step, curr_val, next_val, index)
            self.optimizer.step(self.loss_pinn)