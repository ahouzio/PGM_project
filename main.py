import argparse
from train import train_ncsn, train_sn
from test import test_ncsn, inpaint_ncsn
from load_data import load_dataset
import torch
from models import *
import logging
import torch.distributions as TD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import plot_score_function, batch_jacobian, distribution2score
import os
logging.basicConfig(level=logging.INFO)
logg = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Geneerate samples by estimating the gradient of the data distribution"
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
    parser.add_argument("--model_name", type=str, default="simple", help="model name")
    parser.add_argument("--eps", type=float, default=5e-5, help="eps")
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--directions", type=str, default='right')
    parser.add_argument("--loss_type", type=str, default='denoising')
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
    print("sigmas: ", sigmas)
    # choose dataset
    # print("args.dataset_name ", args.dataset_name)
    # if args.mode == "train" or args.mode == "inpaint":
    if args.dataset == "mnist":
        train_data, test_data = load_dataset(
            args.dataset, flatten=False, binarize=False
        )
        path = './mnist.pth'
    elif args.dataset == "cifar10":
        train_data, test_data = load_dataset(
            args.dataset, flatten=False, binarize=False
        )
        path = './pretrained_models/cifar10.pth'
    elif args.dataset == "celeba":
        train_data, test_data = load_dataset(
            args.dataset, flatten=False, binarize=False
        )
        path = './pretrained_models/celeba.pth'
    elif args.dataset == "mixture" :
        p = 0.2
        noise = 0.1
        mix = TD.Categorical(torch.tensor([p, 1-p]).cuda())
        mv_normals = TD.MultivariateNormal(
            torch.tensor([[1., 1.], [-1., -1.]]).cuda(),
            noise * torch.eye(2).unsqueeze(0).cuda()
        )
        mixture = TD.MixtureSameFamily(mix, mv_normals)
       
        train_data, test_data = train_test_split(mixture.sample((10000,)))
        
    # choose model
    if args.model_name == "ncsn":
        model = NoiseConditionalScoreNetwork(hidden_dim=512)
    elif args.model_name == "condrefinenet":
        model = CondRefineNetDilated()
    elif args.model_name == "simple":
        model = SimpleScoreNetwork(hidden_dim=512)
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
        if args.dataset == "mixture":
            model, batch_loss = train_sn(model, 
                                train_loader=train_loader,
                                n_epochs=n_epochs,
                                sigmas=torch.Tensor([0.1]),
                                lr=args.lr, 
                                conditional=False,
                                use_cuda=args.use_cuda)
        else :
            batch_loss, model = train_ncsn(model, 
                                train_loader,
                                n_epochs,
                                args.lr,
                                sigmas, 
                                args.use_cuda,
                                loss_type=args.loss_type)
        # save model in trained_models folder ceate it if not exist
        model_name = "{args.dataset}_{args.model_name}_{args.loss_type}"
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        torch.save(model, f"trained_models/{model_name}.pth")
        
        # save plot of loss, distribution, score function in traininig_experiments folder
        if not os.path.exists('training_experiments'):
            os.makedirs('training_experiments')
        # get folder exp_i in training_experiments with i is the number of experiments
        exp_i = 0
        while os.path.exists(f'training_experiments/exp_{exp_i}'):
            exp_i += 1
        os.makedirs(f'training_experiments/exp_{exp_i}')
        # save a config file with parameters of the experiment
        with open(f'training_experiments/exp_{exp_i}/config.txt', 'w') as f:
            for key, value in args.__dict__.items():
                f.write('%s:%s\n' % (key, value))
        # save plot of loss after each epoch
        plt.plot(batch_loss['loss'])
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'training_experiments/exp_{exp_i}/loss.png')
        # save plot of distribution,score
        fig,ax = plt.subplots(1,3,figsize=(12,4))
        plot_score_function(distribution2score(mixture), train_data, 'TwoGaussMixture Score Function', ax=ax[0], npts=30)
        samples =  model.sample(n_samples=1000, n_steps=args.n_steps,eps= args.eps)
        plot_score_function(model, samples, 'Predicted scores', ax=ax[1], npts=30, plot_scatter=False)
        plot_score_function(model, samples, 'Predicted scores and samples', ax=ax[2], npts=30)
        plt.savefig(f'training_experiments/exp_{exp_i}/score.png')
    elif args.mode == "test":
        logg.info("Starting testing")
        test_ncsn(
            path=path,
            sigmas=sigmas,
            visualize=True,
            use_cuda=args.use_cuda,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            save_freq=args.save_freq,
            eps=args.eps,
            dataset=args.dataset
        )
    elif args.mode == "inpaint":
        logg.info("Starting inpainting")
        inpaint_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(test_data)),
            batch_size=batch_size,
            shuffle=True
        )
        inpaint_ncsn(
            path=path,
            sigmas=sigmas,
            use_cuda=args.use_cuda,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            dataset=args.dataset,
            direction=args.directions
        )
    else:
        raise ValueError('The argument mode must have the values "train" or "generate"')
