import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import neucube
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def spike_raster(spike_activity, fig_size=(10, 5)):
    """
    Generate a raster plot of spike activity.

    Parameters:
    spike_activity (torch.Tensor): 2D tensor representing spike activity, where each column corresponds to a neuron and each row corresponds to a time step.
    fig_size (tuple, optional): A tuple specifying the figure size. Default is (10, 5).

    Raises:
    ValueError: If the input is not a 2D tensor.

    Returns:
    None
    """

    # Check if input is a 2D tensor
    if not isinstance(spike_activity, torch.Tensor) or len(spike_activity.shape) != 2:
        raise ValueError('Input should be a 2D tensor')
    
    # Find indices of non-zero elements (spikes)
    x, y = torch.where(spike_activity)
    
    # Create scatter plot
    plt.figure(figsize=fig_size)
    plt.scatter(x, y, marker='o', s=1)
    plt.xlabel('Time Steps')
    plt.ylabel('Neuron Index')
    plt.show()

def plot_connections(reservoir, exci_thres=0.002, inhi_thres=-0.003, highlight=False, percentile=75, fig_size=(10, 8), title=None):
    """
    Plot the connections within a NeuCube reservoir.

    Parameters:
    - reservoir (neucube.Reservoir): NeuCube Reservoir object.
    - exci_thres (float): Excitatory threshold for plotting connections. Default is 0.002.
    - inhi_thres (float): Inhibitory threshold for plotting connections. Default is -0.003.
    - highlight (bool): Plot connectivity with values belonging to a given percentile. Default is False.
    - percentile (int): Percentile value used if highlight is True. Default is 75.
    - fig_size (tuple): Figure size (width, height) in inches. Default is (10, 8).
    - title (str): Figure title. Default is None.

    Raises:
    - ValueError: If the input is not a NeuCube Reservoir object.
    """

    # Check if the input is a NeuCube Reservoir object
    if not isinstance(reservoir, neucube.Reservoir):
        raise ValueError('Input should be a NeuCube Reservoir object')

    if highlight:
        pos_weights = reservoir.w_latent_[torch.where(reservoir.w_latent_ > 0.)].cpu().numpy()
        neg_weights = reservoir.w_latent_[torch.where(reservoir.w_latent_ < 0.)].cpu().numpy()
        exci_thres = np.percentile(pos_weights, percentile)
        inhi_thres = np.percentile(neg_weights, 100-percentile)

    exci_vis_conn = torch.where(reservoir.w_latent_ > exci_thres)
    inh_vis_conn = torch.where((reservoir.w_latent_ != 0) & (reservoir.w_latent_ < inhi_thres))

    # Get positions of parent and child neurons for excitatory connections
    exci_parent, exci_children = reservoir.pos_[exci_vis_conn[0]], reservoir.pos_[exci_vis_conn[1]]
    # Get positions of parent and child neurons for inhibitory connections
    inh_parent, inh_children = reservoir.pos_[inh_vis_conn[0]], reservoir.pos_[inh_vis_conn[1]]

    exci_x, exci_b, exci_c = [torch.hstack((exci_parent[:,[i]], exci_children[:,[i]])).cpu() for i in range(3)] 
    inhi_a, inhi_b, inhi_c = [torch.hstack((inh_parent[:,[i]], inh_children[:,[i]])).cpu() for i in range(3)] 

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*reservoir.pos_.T.cpu(), c='black', marker='o')

    exci_lines = np.stack([exci_x, exci_b, exci_c], axis=-1)
    ax.add_collection3d(Line3DCollection(exci_lines, colors='blue'))
    inhi_lines = np.stack([inhi_a, inhi_b, inhi_c], axis=-1)
    ax.add_collection3d(Line3DCollection(inhi_lines, colors='red'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title is not None:
        ax.set_title(title)
    plt.show()
    
def plot_connections_detailed(reservoir, sup_title=None, highlight=False, percentile=75, exci_thres=0.002, inhi_thres=-0.003, fig_size=(12, 12), eeg_coordinates=None):
    """
    Plot the connections within a NeuCube reservoir.

    Parameters:
    - reservoir (neucube.Reservoir): NeuCube Reservoir object.
    - sup_title (str): Figure title. Default is None.
    - highlight (bool): Plot connectivity with values belonging to a given percentile. Default is False.
    - percentile (int): Percentile value used if highlight is True. Default is 75.
    - exci_thres (float): Excitatory threshold for plotting connections, overriden if highlight is True. Default is 0.002.
    - inhi_thres (float): Inhibitory threshold for plotting connections, overriden if highlight is True. Default is -0.003.
    - fig_size (tuple): Figure size (width, height) in inches. Default is (12, 12).
    - eeg_coordinates (dict): Dictionary consisting of input neurons and corresponding text label. Default is None. 

    Raises:
    - ValueError: If the input is not a NeuCube Reservoir object and/or if eeg_coordinates is not provided.
    """

    # Check if the input is a NeuCube Reservoir object
    if not isinstance(reservoir, neucube.Reservoir):
        raise ValueError('Input should be a NeuCube Reservoir object')

    if eeg_coordinates is None:
        raise ValueError("EEG Info must be provided")
    
    
    pos_weights = reservoir.w_latent_[torch.where(reservoir.w_latent_ > 0.)].cpu().numpy()
    neg_weights = reservoir.w_latent_[torch.where(reservoir.w_latent_ < 0.)].cpu().numpy()
    pos_avg, pos_std = pos_weights.mean(), np.std(pos_weights)
    neg_avg, neg_std = neg_weights.mean(), np.std(neg_weights)
    
    
    if highlight:
        exci_thres = np.percentile(pos_weights, percentile)
        inhi_thres = np.percentile(neg_weights, 100-percentile)
    
    exci_vis_conn = torch.where(reservoir.w_latent_ > exci_thres)
    inh_vis_conn = torch.where((reservoir.w_latent_ != 0) & (reservoir.w_latent_ < inhi_thres))

    # Get positions of parent and child neurons for excitatory connections
    exci_parent, exci_children = reservoir.pos_[exci_vis_conn[0]], reservoir.pos_[exci_vis_conn[1]]
    # Get positions of parent and child neurons for inhibitory connections
    inh_parent, inh_children = reservoir.pos_[inh_vis_conn[0]], reservoir.pos_[inh_vis_conn[1]]

    exci_x, exci_b, exci_c = [torch.hstack((exci_parent[:,[i]], exci_children[:,[i]])).cpu() for i in range(3)] 
    inhi_a, inhi_b, inhi_c = [torch.hstack((inh_parent[:,[i]], inh_children[:,[i]])).cpu() for i in range(3)] 

    fig, axs = plt.subplots(2, 2, figsize=fig_size, subplot_kw=dict(projection='3d'))
    

    
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    
    axes = [ax1, ax2, ax3, ax4]    
    
    for ax in axes:
        ax.scatter(*reservoir.pos_.T.cpu(), c='black', marker='o', )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    axes[0].set_title("Input Connections")
    for channel in eeg_coordinates:
        axes[0].scatter(*eeg_coordinates[channel], c='green', marker='s')
        zdir = (eeg_coordinates[channel][0] + 1, eeg_coordinates[channel][1] + 1, eeg_coordinates[channel][2] + 1)
        axes[0].text(eeg_coordinates[channel][0], eeg_coordinates[channel][1], eeg_coordinates[channel][2], channel, zdir, bbox=dict(facecolor='red', alpha=0.5))
    
    axes[1].set_title("All Connections")
    exci_lines = np.stack([exci_x, exci_b, exci_c], axis=-1)
    axes[1].add_collection3d(Line3DCollection(exci_lines, colors='blue'))
    inhi_lines = np.stack([inhi_a, inhi_b, inhi_c], axis=-1)
    axes[1].add_collection3d(Line3DCollection(inhi_lines, colors='red'))
    axes[1].legend(handles=[Line2D([0], [0], color='red', lw=3, label='Negative Connection'),
                     Line2D([0], [0], color='blue', lw=3, label='Positive Connection')])
    
    axes[2].set_title("Excitatory Connections")
    axes[2].add_collection3d(Line3DCollection(exci_lines, colors='blue'))
    axes[2].legend(handles=[Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=f'Positive Weights: {pos_weights.shape[0]}'),
                     Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=(f'Mean (std): {"{:.2f}".format(pos_avg)} ({"{:.2f}".format(pos_std)})')),
                     Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=f'{percentile} Percentile: {"{:.2f}".format(exci_thres)}')])

    axes[3].set_title("Inhibitory Connections")
    axes[3].add_collection3d(Line3DCollection(inhi_lines, colors='red'))
    axes[3].legend(handles=[Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=f'Negative Weights: {neg_weights.shape[0]}'),
                     Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=f'Mean (std): {"{:.2f}".format(neg_avg)} ({"{:.2f}".format(neg_std)})'),
                     Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label=f'{100-percentile} Percentile: {"{:.2f}".format(inhi_thres)}')])
    

    
    if type(sup_title) is str:
        plt.suptitle(sup_title)
    
    plt.show()