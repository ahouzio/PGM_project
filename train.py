import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def train_ncsn(
    model: object,
    train_loader: object,
    n_epochs: int,
    lr: float,
    sigmas: torch.Tensor,
    use_cuda: bool = False,
    loss_type: str = "denoising",
) -> dict:
    if use_cuda:
        model = model.cuda()
    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))
    batch_loss_history = {"loss": []}
    for epoch_i in tqdm(range(n_epochs)):
        print("len(train_loader) ", len(train_loader))
        for batch_i, x in tqdm(enumerate(train_loader)):
            x = x[0]
            batch_size = x.shape[0]
            if use_cuda:
                x = x.cuda()

            labels = torch.randint(len(sigmas), (batch_size,))
            sigma_batch = sigmas[labels].to(x.device)
            # sigma_batch = sigmas[torch.randint(len(sigmas), (batch_size,))].to(x.device)
            sigma_batch = sigma_batch.reshape(-1, *([1] * (len(x.shape) - 1)))
            if loss_type == "denoising":
                standard_noise = torch.randn_like(x)
                x_noisy = x + standard_noise * sigma_batch

                optimizer.zero_grad()
                # pred_scores = model(x_noisy, sigma_batch.flatten())
                pred_scores = model(x_noisy, labels.to(x.device))

                noisy_scores = (-standard_noise / sigma_batch).reshape(batch_size, -1)
                pred_scores = pred_scores.reshape(batch_size, -1)

                losses = torch.sum((pred_scores - noisy_scores) ** 2, axis=-1) / 2
                loss = torch.mean(losses * sigma_batch.flatten() ** 2)
            elif loss_type == "sciced_score_matching":
                loss = sliced_score_matchng(model, x, sigma_batch, labels)
            loss.backward()
            optimizer.step()

            batch_loss_history["loss"].append(loss.data.cpu().numpy())

        # save batch_loss_history n a file
        with open('batch_loss_history.txt', 'w') as f:
            for key, value in batch_loss_history.items():
                f.write('%s:%s\n' % (key, value))
    return batch_loss_history

# Implement from the paper "Sliced Score Matching: A Scalable Approach to Density and Score Estimation"
def sliced_score_matchng(model, x, sigma, labels, M, distribution):
    """
    Input: ˜pm(·; θ), x, v
    1: sm(x; θ) ← grad(log ˜pm(x; θ), x)
    2: vᵀ∇xsm(x; θ) ← grad(vᵀsm(x; θ), x)
    3: J ← 1
    2 (vᵀsm(x; θ))2 (or J ← 1
    2 ‖sm(x; θ)‖2
    2)
    4: J ← J + vᵀ∇xsm(x; θ)v
    return J
    Args:
        direction (_type_): 
        model (_type_): _description_
        x (_type_): _description_
        sigma (_type_): _description_
        n_steps_each (_type_): _description_
        step_lr (_type_): _description_
        img_size (_type_): _description_
        n_steps (_type_): _description_
        use_cuda (_type_): _description_
    """
    sm = model(x, labels)
    # M directions
    if distribution == "normal":
        v = torch.randn(M, *x.shape).to(x.device)
    elif distribution == "rademacher":
        v = torch.randint(0, 2, (M, *x.shape)).to(x.device) * 2 - 1
    # grad(vᵀsm(x; θ), x)
    N = x.shape[0]
    J = 0 # loss
    for i in range(N): # N number of samples in the baych
        for j in range(M): # M number of directions
            sm = model(x[i], labels[i]) # h(x_i, theta)
            # compute the gradient with rrespect to x grad(vᵀsm(x; θ), x)
            grad_sm = torch.autograd.grad(sm, x[i], create_graph=True)[0]
            J += 0.5 * (grad_sm * v[j]).sum() ** 2 + 0.5 * torch.sum(v[j] * sm) ** 2
            
    return J / (N * M)
            
            
            
    
    
    
    
    
    
 

def save_model(model: object, optimizer:object, path: str) -> None:
    torch.save([model.state_dict(), optimizer.state_dict()], path)
