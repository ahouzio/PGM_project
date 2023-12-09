import argparse
from train import train_ncsn
from test import test_ncsn, inpaint_ncsn
from load_data import load_dataset
import torch
from models import *
import logging
logging.basicConfig(level=logging.INFO)
logg = logging.getLogger(__name__)

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
    parser.add_argument("--use_cuda", type=bool, default=True, help="use cuda")
    parser.add_argument("--mode", type=str, default="train", help="mode")
    parser.add_argument("--n_samples", type=int, default=5, help="number of samples")
    parser.add_argument("--n_steps", type=int, default=100, help="number of steps")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--model_name", type=str, default="ncsn", help="model name")
    parser.add_argument("--path", type=str, default="./pretrained_models/mnist.pth", help="path to model")
    parser.add_argument("--eps", type=float, default=5e-5, help="eps")
    args = parser.parse_args()
    
    # command to train the model will be like:
    # python main.py --dataset_name mnist --batch_size 32 --n_epochs 10 --lr 1e-4 --lambda_max 0.01 --lambda_min 1e-4 --n_lambdas 10 --use_cuda True --mode train --model_name ncsn

    return args


if __name__ == "__main__":
    args = main()
    batch_size = args.batch_size
    
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
    elif args.model_name == "simple":
        model = SimpleScoreNetwork()
    else:
        raise ValueError(
            'The argument model_name must have the values "ncsn", "condrefinenet" or "refinenet"'
        )

    # train or test
    if args.mode == "train":
        logg.info("Starting training")
        n_epochs = args.n_epochs
        lr = args.lr
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_data)),
            batch_size=batch_size,
            shuffle=True,
        )
        train_ncsn(model, train_loader, n_epochs, args.lr, sigmas, args.use_cuda)

    elif args.mode == "test":
        logg.info("Starting testing")
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(test_data)),
            batch_size=batch_size,
            shuffle=True
        )
        print("data loaded")
        test_ncsn(
            path=args.path,
            sigmas=sigmas,
            visualize=True,
            use_cuda=args.use_cuda,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            save_freq=args.save_freq,
            eps=args.eps,
        )
    elif args.mode == "inpaint":
        logg.info("Starting inpainting")
        inpaint_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(test_data)),
            batch_size=batch_size,
            shuffle=True
        )
        inpaint_ncsn(
            path=args.path,
            sigmas=sigmas,
            use_cuda=args.use_cuda,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
        )
    else:
        raise ValueError('The argument mode must have the values "train" or "generate"')
