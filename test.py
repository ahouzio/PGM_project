import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from models import NoiseConditionalScoreNetwork
import torch
from torchvision.utils import make_grid
import os
from load_data import load_dataset
from PIL import Image

def test_ncsn(
    path: str,
    sigmas: torch.Tensor,
    visualize: bool = True,
    use_cuda: bool = False,
    n_samples: int = 5,
    n_steps: int = 100,
    save_freq: int = 50,
):
    refine_net = NoiseConditionalScoreNetwork(use_cuda=use_cuda)
    refine_net.load_state_dict(torch.load(path))
    refine_net.cuda()
    refine_net.eval()
    samples, history = refine_net.sample(
        n_samples=n_samples, n_steps=n_steps, sigmas=sigmas, save_history=True
    )
    if visualize:
        visualize_history(samples, history, sigmas, save_freq, save_folder=f"{n_samples}_samples_{n_steps}_steps")

def visualize_history(samples, history,sigmas, save_freq, save_folder="samples"):
    print("Visualizing history")
    grid_samples = make_grid(samples, nrow=5)
    grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)  
    print("Saving images")
    # creae save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    steps_per_sigma = int(len(history) / len(sigmas))   
    for step in range(len(history)):
        sigma_step = step // steps_per_sigma
        grid_samples = make_grid(history[step], nrow=5)
        grid_img = grid_samples.permute(1, 2, 0).clip(0, 1)
        # save images in the save folder after converting them to numpy arrays
        grid_img = grid_img.cpu().numpy()
        step_size = sigma_step * save_freq
        plt.imsave(f"{save_folder}/sigma_{sigmas[sigma_step]:.2f}_step_{step_size}_.png", grid_img)


def inpaint_ncsn(
            path,
            sigmas,
            visualize,
            use_cuda
        ) :
    n_samples = 5
    n_steps = 100
    refine_net = NoiseConditionalScoreNetwork(use_cuda=use_cuda)
    refine_net.load_state_dict(torch.load(path))
    refine_net.cuda()
    refine_net.eval()
    # download test samples of MNIST
    train_data, test_data = load_dataset("mnist", flatten=False, binarize=False)
    test_data = test_data[:n_samples]
    # crop a random part of each test sample
    cropped_test_data = []
    print("len(test_data): ", len(test_data))
    w,h = test_data[0].shape[1], test_data[0].shape[2]
    print("width and height of the test data: ", w, h)
    for i in range(len(test_data)):
        cropped_x = np.random.randint(0, w//2)
        cropped_y = np.random.randint(0, h//2)
        # copy original image
        cropped_img = test_data[i].copy()
        # remove the cropped part from the image by setting it to zero
        cropped_img[:, cropped_x:cropped_x+w//2, cropped_y:cropped_y+h//2] = 0
        cropped_test_data.append(cropped_img) 
        
    # sample  using croppedd image as input
    cropped_test_data = torch.tensor(cropped_test_data)
    cropped_test_data = cropped_test_data.cuda()
    samples, history = refine_net.sample(
        n_samples=n_samples,
        n_steps=n_steps,
        sigmas=sigmas,
        save_history=True,
        init_samples=cropped_test_data.clone(),
    )
    # save the original, cropped and inpainted images in one image each
    # create save folder
    if not os.path.exists("inpainting"):
        os.makedirs("inpainting")
    mosaic_list = []
    for i in range(len(test_data)):
        original_img = torch.tensor(test_data[i]).cuda()
        cropped_img = cropped_test_data[i].cuda()
        inpainted_img = samples[i].cuda()
        inpainted_img = inpainted_img.clip(0, 1)
        original_img = original_img.cpu().numpy()
        cropped_img = cropped_img.cpu().numpy()
        inpainted_img = inpainted_img.cpu().numpy()
        mosaic = np.concatenate([original_img,cropped_img, inpainted_img], axis=2)[0]
        # add a white line between the images
        mosaic = np.concatenate([mosaic, np.ones((1,mosaic.shape[1]))], axis=0)
        mosaic_list.append(mosaic)
    mosaic = np.concatenate(mosaic_list, axis=0)
    plt.imsave("inpainting/mosaic.png", mosaic, cmap="gray")
    print("mosaic saved")
        
    
    
           
