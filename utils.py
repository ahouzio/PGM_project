import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()

def plot_score_function(
    score_function,
    data,
    title,
    plot_scatter=True,
    xlim=(-2., 2.),
    ylim=(-2., 2.),
    npts=40,
    ax=None,
    figsize=(12, 12),
    scatter_label='GT labels',
    quiver_label = None,
    quiver_color='black'
):
    xx = np.stack(np.meshgrid(np.linspace(xlim[0], xlim[1], npts),
                              np.linspace(ylim[0], ylim[1], npts)), axis=-1).reshape(-1, 2)

    input = torch.tensor(xx).float()
    if USE_CUDA:
      scores = score_function(input.cuda()).detach().cpu().numpy()
    else:
      scores = score_function(input).detach().numpy()

    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Draw the plots
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if plot_scatter:
        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, color='blue', s=5, label=scatter_label)
    ax.quiver(*xx.T, *scores_log1p.T, width=0.002, color=quiver_color, label=quiver_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=16)
    return ax

def batch_jacobian(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape_inp)
    output: tensor (bs, *shape_oup) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape_oup, *shape_inp)
    '''
    def out_permutation():
        n_inp = np.arange(len(input.shape) - 1)
        n_output = np.arange(len(output.shape) - 1)
        return tuple(np.concatenate([n_output + 1, [0,], n_inp + len(n_output) + 1]))

    s_output = torch.sum(output, dim=0) # sum by batch dimension
    batched_grad_outputs = torch.eye(
        np.prod(s_output.shape)).view((-1,) + s_output.shape).to(output)
    # batched_grad_outputs = torch.eye(s_output.size(0)).to(output)
    grad = autograd.grad(
        outputs=s_output, inputs=input,
        grad_outputs=batched_grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
        is_grads_batched=True
    )
    return grad[0].permute(out_permutation())


def distribution2score(distribution):
    def score_function(x):
        x.requires_grad_(True) # (bs, 2)
        log_prob = distribution.log_prob(x).unsqueeze(-1) # (bs, 1)
        s_raw = batch_jacobian(
            x, log_prob, create_graph=False, retain_graph=False) # (bs, 1, 2)
        return s_raw.reshape(x.size(0), -1).detach()
    return score_function



    
             