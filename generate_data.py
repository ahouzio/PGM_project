

# generate a 2D mixture of Gaussians dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from models import NoiseConditionalScoreNetwork
import torch.distributions as TD
    
USE_CUDA = torch.cuda.is_available()

def train_test_split(data):
    count = data.shape[0]
    split = int(0.8 * count)
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data

def create_two_gaussians(p, noise=0.1):
    mix = TD.Categorical(torch.tensor([p, 1 - p]))
    mv_normals = TD.MultivariateNormal(
        torch.tensor([[1., 1.], [-1., -1.]]),
        noise * torch.eye(2).unsqueeze(0).repeat_interleave(2, 0))

    if USE_CUDA:
      return TD.MixtureSameFamily(mix, mv_normals).cuda()

    return TD.MixtureSameFamily(mix, mv_normals)
