import argparse
from train import train_ncsn
from test import test_ncsn
from load_data import load_dataset
import torch
from models import *


def main():
    parser = argparse.ArgumentParser(
        description="Geneerate samples by estimating the gradient of the data distribution"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mnist", help="dataset name"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lambda_max", type=float, default=0.01, help="lambda max")
    parser.add_argument("--lambda_min", type=float, default=1e-4, help="lambda min")
    parser.add_argument("--n_lambdas", type=int, default=10, help="number of lambdas")
    parser.add_argument("--use_cuda", type=bool, default=False, help="use cuda")
    parser.add_argument("--mode", type=str, default="train", help="mode")
    parser.add_argument("--model_name", type=str, default="ncsn", help="model name")
    args = parser.parse_args()
    
    # command to train the model will be like:
    # python main.py --dataset_name mnist --batch_size 128 --n_epochs 100 --lr 1e-4 --lambda_max 0.01 --lambda_min 1e-4 --n_lambdas 10 --use_cuda False --mode train --model_name ncsn

    return args


if __name__ == "__main__":
    args = main()
    # check cuda
    if args.use_cuda:
        assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.empty_cache()
    # create sigmas
    sigmas = torch.tensor(
        np.exp(
            np.linspace(
                np.log(args.lambda_max), np.log(args.lambda_min), args.n_lambdas
            )
        ),
        dtype=torch.float32,
    )
    # choose dataset
    if args.dataset_name == "mnist":
        train_data, test_data = load_dataset(
            args.dataset_name, flatten=False, binarize=False
        )
    elif args.dataset_name == "cifar10":
        train_data, test_data = load_dataset(
            args.dataset_name, flatten=False, binarize=False
        )
    else:
        raise ValueError('The argument dataset_name must have the values "mnist" or "cifar10"')

    # choose model
    if args.model_name == "ncsn":
        model = NoiseConditionalScoreNetwork()
    elif args.model_name == "condrefinenet":
        model = CondRefineNetDilated()
    else:
        raise ValueError(
            'The argument model_name must have the values "ncsn", "condrefinenet" or "refinenet"'
        )

    # train or test
    if args.mode == "train":
        n_epochs = args.n_epochs
        lr = args.lr
        batch_size = args.batch_size
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_data)),
            batch_size=batch_size,
            shuffle=True,
        )
        train_ncsn(model, train_loader, n_epochs, args.lr, sigmas, args.use_cuda)

    elif args.mode == "test":
        test_loader = torch.utils.data.DataLoader(
            path=args.path,
            sigmas=sigmas,
            visualize=True,
            print_loss=True,
        )
        test_ncsn(model, test_loader, args.n_epochs, args.lr, use_cuda=args.use_cuda)
    else:
        raise ValueError('The argument mode must have the values "train" or "generate"')
