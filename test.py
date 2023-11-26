import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from models import NoiseConditionalScoreNetwork
import torch
from torchvision.utils import make_grid


def test_ncsn(
    path: str,
    sigmas: torch.Tensor,
    visualize: bool = False,
    print_loss: bool = False,
):

    refine_net = NoiseConditionalScoreNetwork()
    refine_net.load_state_dict(torch.load(path))
    refine_net.cuda()
    refine_net.eval()
    samples, history = refine_net.sample(
        n_samples=10, n_steps=100, sigmas=sigmas, save_history=True
    )
    if print_loss:
        print(history)
    if visualize:
        visualize_history(samples, history)


def visualize_history(samples, history):
    grid_samples = make_grid(samples, nrow=5)

    grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
    # plt.figure(figsize=(6, 6))
    plt.imshow(grid_img, cmap="Greys")
    plt.axis("off")
    plt.show()

    for step in range(len(history)):
        grid_samples = make_grid(history[step], nrow=5)

        grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
        # plt.figure(figsize=(6, 6))
        plt.imshow(grid_img, cmap="Greys")
        plt.axis("off")
        plt.show()
