import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import pathlib

def create_plot(data, x, t, cmap, xlabel, ylabel, file_path, vmin=None, vmax=None):
    """Helper function to create and save a plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    h = ax.imshow(data, interpolation='nearest', cmap=cmap,
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    
    ax.set_xlabel(xlabel, size=15)
    ax.set_ylabel(ylabel, size=15)
    ax.tick_params(labelsize=15)

    plt.savefig(file_path)
    plt.close()

def u_exact(Exact, x, t, cfgs, file_path):
    """Visualize exact solution."""
    create_plot(Exact.T, x, t, 'rainbow', 't', 'x', file_path)
    
    return None

def u_diff(Exact, U_pred, x, t, cfgs, file_path, relative_error=False):
    """Visualize abs(u_pred - u_exact)."""
    diff = np.abs(Exact.T - U_pred.T)
    
    if relative_error:
        diff /= np.abs(Exact.T)
    
    create_plot(diff, x, t, 'binary', 't', 'x', file_path)

    return None

def u_predict(U_vals, U_pred, x, t, cfgs, file_path):
    """Visualize u_predicted."""
    create_plot(U_pred.T, x, t, 'rainbow', 't', 'x', file_path, vmin=U_vals.min(), vmax=U_vals.max())

    return None
