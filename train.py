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
) -> dict:
    if use_cuda:
        model = model.cuda()
    model.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))
    batch_loss_history = {"loss": []}
    for epoch_i in tqdm(range(n_epochs)):
        print("len(train_loader) ", len(train_loader))
        for batch_i, x in enumerate(train_loader):
            print("batch_i ", batch_i)
            x = x[0]
            batch_size = x.shape[0]
            if use_cuda:
                x = x.cuda()

            labels = torch.randint(len(sigmas), (batch_size,))
            sigma_batch = sigmas[labels].to(x.device)
            # sigma_batch = sigmas[torch.randint(len(sigmas), (batch_size,))].to(x.device)
            sigma_batch = sigma_batch.reshape(-1, *([1] * (len(x.shape) - 1)))
            standart_noise = torch.randn_like(x)
            x_noisy = x + standart_noise * sigma_batch

            optimizer.zero_grad()
            # pred_scores = model(x_noisy, sigma_batch.flatten())
            pred_scores = model(x_noisy, labels.to(x.device))

            noisy_scores = (-standart_noise / sigma_batch).reshape(batch_size, -1)
            pred_scores = pred_scores.reshape(batch_size, -1)

            losses = torch.sum((pred_scores - noisy_scores) ** 2, axis=-1) / 2
            loss = torch.mean(losses * sigma_batch.flatten() ** 2)

            loss.backward()
            optimizer.step()

            batch_loss_history["loss"].append(loss.data.cpu().numpy())

        # save batch_loss_history n a file
        with open('batch_loss_history.txt', 'w') as f:
            for key, value in batch_loss_history.items():
                f.write('%s:%s\n' % (key, value))
    return batch_loss_history

def save_model(model: object, optimizer:object, path: str) -> None:
    torch.save([model.state_dict(), optimizer.state_dict()], path)
