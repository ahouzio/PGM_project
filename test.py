import numpy as np
import matplotlib.pyplot as plt
from models import NoiseConditionalScoreNetwork
import torch
import os
from load_data import load_dataset
import gc
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
from utils import distribution2score


def test_ncsn(
    path: str,
    sigmas: torch.Tensor,
    visualize: bool = True,
    use_cuda: bool = False,
    n_samples: int = 5,
    n_steps: int = 100,
    save_freq: int = 50,
    eps: float = 5e-5,
    dataset: str = "mnist",
):
    if dataset == "mnist":
        refine_net = NoiseConditionalScoreNetwork(use_cuda=use_cuda)
    elif dataset == "cifar10":
        print("dataset is cifar10")
        refine_net = NoiseConditionalScoreNetwork(
            use_cuda=use_cuda, n_channels=3, image_size=32, num_classes=10, ngf=128
        )
    elif dataset == "celeba":
        refine_net = NoiseConditionalScoreNetwork(
            use_cuda=use_cuda, n_channels=3, image_size=32, num_classes=10, ngf=128
        )
    print("dataset: ", dataset)
    print("path: ", path)
    states = torch.load(path)
    pretrained = False
    if len(states) == 2:  # optimizer state was also saved in the checkpoint
        refine_net.load_state_dict(states[0])
        pretrained = True
    else:
        refine_net.load_state_dict(torch.load(path))
    print("Model is pretrained: ", pretrained)
    refine_net.cuda()
    refine_net.eval()
    samples, history = refine_net.sample(
        n_samples=n_samples, n_steps=n_steps, sigmas=sigmas, eps=eps, save_history=True
    )
    if visualize:
        visualize_history(
            samples,
            history,
            sigmas,
            save_freq,
            pretrained,
            dataset=dataset,
            save_folder=f"{n_samples}_samples_{n_steps}_steps_sigma_{sigmas[0]:.4f}_{sigmas[-1]:.4f}_eps_{eps:.5f}_dataset_{dataset}",
        )


def test_mix(mixture, test_data: torch.Tensor, sigmas: torch.Tensor):

    true_scores = distribution2score(mixture)(test_data.cuda(), None)
    labels = torch.arange(len(sigmas)).cuda()
    labels = labels.repeat_interleave(test_data.size(0) // len(labels))
    predicted_losses = []
    for model_name in os.listdir("trained_models"):
        model = torch.load(f"trained_models/{model_name}")
        score = model(test_data.cuda(), labels)
        loss = 0.5*(torch.norm(score - true_scores, p=2, dim=-1)**2).mean()
        predicted_losses.append(loss.detach().cpu().numpy())
    return predicted_losses

def visualize_history(
    samples, history, sigmas, save_freq, pretrained, dataset, save_folder="samples"
):
    print("Visualizing history")
    grid_samples = make_grid(samples, nrow=5)
    grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
    print("Saving images")
    # creae save folder
    if not os.path.exists(save_folder):
        if pretrained:
            save_folder = save_folder + "_pretrained"
            os.makedirs(save_folder)
        else:
            os.makedirs(save_folder)
    steps_per_sigma = int(len(history) / len(sigmas))
    for step in range(len(history)):
        sigma_step = step % steps_per_sigma
        sigma_idx = step // steps_per_sigma
        grid_samples = make_grid(history[step], nrow=5)
        grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
        # save images in the save folder after converting them to numpy arrays
        grid_img = grid_img.cpu().numpy()
        step_size = sigma_step * save_freq
        print("grid img min max: ", grid_img.min(), grid_img.max())
        plt.imsave(
            f"{save_folder}/sigma_{sigmas[sigma_idx]:.4f}_step_{step_size}.png",
            grid_img,
        )
    gc.collect()


def anneal_Langevin_dynamics_inpainting(
    x_mod,
    original_image,
    scorenet,
    sigmas,
    img_size,
    n_channels,
    direction="left",
    n_steps_each=100,
    step_lr=0.000008,
):
    images = []
    original_image = original_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    original_image = original_image.contiguous().view(
        -1, n_channels, img_size, img_size
    )
    x_mod = x_mod.view(-1, n_channels, img_size, img_size)
    if direction == "left":
        half_original_image = original_image[:, :, :, : img_size // 2]
    elif direction == "right":
        half_original_image = original_image[:, :, :, img_size // 2 :]
    elif direction == "top":
        half_original_image = original_image[:, :, : img_size // 2, :]
    elif direction == "bottom":
        half_original_image = original_image[:, :, img_size // 2 :, :]
    # save half original image
    # save_image(half_original_image, 'inpainting/half_original_image_
    with torch.no_grad():
        for c, sigma in tqdm(
            enumerate(sigmas),
            total=len(sigmas),
            desc="annealed Langevin dynamics sampling",
        ):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            corrupted_half_image = (
                half_original_image + torch.randn_like(half_original_image) * sigma
            )
            # save corrupted half image
            if direction == "left":
                x_mod[:, :, :, : img_size // 2] = corrupted_half_image
            elif direction == "right":
                x_mod[:, :, :, img_size // 2 :] = corrupted_half_image
            elif direction == "top":
                x_mod[:, :, : img_size // 2, :] = corrupted_half_image
            elif direction == "bottom":
                x_mod[:, :, img_size // 2 :, :] = corrupted_half_image

            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to("cpu"))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                if direction == "left":
                    x_mod[:, :, :, : img_size // 2] = corrupted_half_image
                elif direction == "right":
                    x_mod[:, :, :, img_size // 2 :] = corrupted_half_image
                elif direction == "top":
                    x_mod[:, :, : img_size // 2, :] = corrupted_half_image
                elif direction == "bottom":
                    x_mod[:, :, img_size // 2 :, :] = corrupted_half_image
        #
        return images


def inpaint_ncsn(path, sigmas, use_cuda, n_samples, n_steps, dataset, direction):

    if dataset == "mnist":
        refine_net = NoiseConditionalScoreNetwork(
            use_cuda=use_cuda, n_channels=1, image_size=28, num_classes=10
        )
    elif dataset == "cifar10":
        refine_net = NoiseConditionalScoreNetwork(
            use_cuda=use_cuda, n_channels=3, image_size=32, num_classes=10, ngf=128
        )
    states = torch.load(path)
    if len(states) == 2:  # optimizer state was also saved in the checkpoint
        refine_net.load_state_dict(states[0])
        pretrained = True
    else:
        refine_net.load_state_dict(torch.load(path))
    refine_net.cuda()
    refine_net.eval()
    # download test samples of MNIST
    if dataset == "mnist":
        train_data, test_data = load_dataset("mnist", flatten=False, binarize=False)
    if dataset == "cifar10":
        train_data, test_data = load_dataset("cifar10", flatten=False, binarize=False)

    data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    test_data = next(iter(data_loader))
    samples = torch.rand(
        n_samples,
        n_samples,
        refine_net.n_channels,
        refine_net.image_size,
        refine_net.image_size,
    ).cuda()
    # save this test
    save_image(
        test_data,
        "inpainting/original_{0}_{1}_{2}_{3}.png".format(
            dataset, direction, n_steps, n_samples
        ),
        nrow=5,
    )
    images = anneal_Langevin_dynamics_inpainting(
        samples,
        test_data,
        refine_net,
        sigmas,
        n_steps_each=n_steps,
        step_lr=0.00002,
        img_size=refine_net.image_size,
        n_channels=refine_net.n_channels,
        direction=direction,
    )
    imgs = []
    for i, sample in tqdm(enumerate(images)):
        sample = sample.view(
            n_samples**2,
            refine_net.n_channels,
            refine_net.image_size,
            refine_net.image_size,
        )
        image_grid = make_grid(sample, nrow=n_samples)
        if i % 10 == 0:
            im = Image.fromarray(
                image_grid.mul_(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            imgs.append(im)
        # save last image
        # save_image(image_grid, 'inpainting/latest_inpainting_{0}_{1}_{2}_{3}_sigma_{4}_{5}_{6}.png'.format(dataset, direction, n_steps, n_samples, sigmas[0], sigmas[-1], i), nrow=n_samples)
    imgs[-1].save(
        "inpainting/latest_inpainting_{0}_{1}_{2}_{3}_sigma_{4}_{5}.png".format(
            dataset, direction, n_steps, n_samples, sigmas[0], sigmas[-1]
        )
    )
    imgs[0].save(
        "inpainting/inpainting_{0}_{1}_{2}_{3}_sigma_{4}_{5}.gif".format(
            dataset, direction, n_steps, n_samples, sigmas[0], sigmas[-1]
        ),
        save_all=True,
        append_images=imgs[1:],
        optimize=False,
        duration=40,
        loop=0,
    )
