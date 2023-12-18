import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from utils import batch_jacobian

def train_sn(
    model: object,
    train_loader: object,
    n_epochs: int,
    lr: float,
    sigmas: torch.Tensor = torch.Tensor([0.1]),
    use_cuda: bool = False,
    conditional: bool = True,
    loss_type: str = "denoising_score_matching",
    n_vectors: int = 1,
    dist_type: str = "normal",
) -> dict:

    if use_cuda:
        critic = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))
    batch_loss_history = {"loss": []}
    
    for epoch_i in tqdm(range(n_epochs)):
        mean_loss = 0
        for batch_i, x in enumerate(train_loader):
            x = x[0]
            batch_size = x.shape[0]
            if use_cuda:
                x = x.cuda()
        
            if conditional:
                labels = torch.randint(len(sigmas), (batch_size,))
                sigma_batch = sigmas[labels].to(x.device)
                sigma_batch = sigma_batch.reshape(-1, 1)
            else:
                sigma_batch = (
                    sigmas[0] * torch.ones(batch_size, 1, device=x.device).float()
                )
            if loss_type == "denoising_score_matching":
                standart_noise = torch.randn_like(x)
                x_noisy = x + standart_noise * sigma_batch
                optimizer.zero_grad()
                if conditional:
                    pred_scores = model(x_noisy, labels.to(x.device))
                else:
                    pred_scores = model(x_noisy)  
                noisy_scores = -standart_noise / sigma_batch
                losses = torch.sum((pred_scores - noisy_scores) ** 2, axis=-1) / 2
                loss = torch.mean(losses * sigma_batch.flatten() ** 2)
            elif loss_type == "sliced_score_matching":
                x.requires_grad_(True)
                optimizer.zero_grad()
                if conditional :
                    loss = sliced_score_matching(model, x, labels, n_vectors, dist_type)
                else:
                    loss = sliced_score_matching(model, x, None, n_vectors, dist_type)
            elif loss_type == "sliced_score_matching_vr":
                x.requires_grad_(True)
                optimizer.zero_grad()
                if conditional :
                    loss = sliced_score_matching_vr(model, x, labels, n_vectors, dist_type)
                else:
                    loss = sliced_score_matching_vr(model, x, None, n_vectors, dist_type)
            loss.backward()
            optimizer.step()
            mean_loss += loss.data.cpu().numpy()
        batch_loss_history["loss"].append(mean_loss / len(train_loader))
    return model, batch_loss_history

# Implement from the paper "Sliced Score Matching: A Scalable Approach to Density and Score Estimation"
def sliced_score_matching(model, x, labels, M, distribution):
    if distribution == "normal":
        v = torch.randn(M, *x.shape).to(x.device)
    elif distribution == "rademacher":
        v = torch.randint(0, 2, (M, *x.shape)).to(x.device) * 2 - 1
    N = x.shape[0]
    J = 0  # loss
    if labels != None:
        sm = model(x, labels.to(x.device))
    else:
        sm = model(x)
    grad_sm = batch_jacobian(input=x, output=sm)
    for i in range(N):
        for j in range(M): 
            J += 0.5 * torch.matmul(torch.matmul(v[j][i], grad_sm[i]), v[j][i]) + 0.5 * torch.matmul(v[j][i], sm[i])**2
    return J / (N * M)

def sliced_score_matching_vr(model, x, labels, M, distribution):
    # print(labels, x.shape, labels.shape)
    # M directions
    if distribution == "normal":
        v = torch.randn(M, *x.shape).to(x.device)
    elif distribution == "rademacher":
        v = torch.randint(0, 2, (M, *x.shape)).to(x.device) * 2 - 1
    N = x.shape[0]
    J = 0  # loss
    if labels != None:
        sm = model(x, labels.to(x.device))
    else:
        sm = model(x)
    grad_sm = batch_jacobian(input=x, output=sm)
    for i in range(N):
        for j in range(M):  
            J += 0.5 * torch.matmul(torch.matmul(v[j][i], grad_sm[i]), v[j][i]) + 0.5 * torch.norm(sm[i], p=2)**2

    return J / (N * M)

def save_model(model: object, optimizer: object, path: str) -> None:
    torch.save([model.state_dict(), optimizer.state_dict()], path)
