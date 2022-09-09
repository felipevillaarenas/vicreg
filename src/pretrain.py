
import os
from pathlib import Path
from argparse import ArgumentParser

from datamodule import PreTrainDataModule
from vicreg import VICReg

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR10, STL10, ImageNet

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

from transforms import VICRegTrainDataTransform, VICRegEvalDataTransform

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], description="Pretrain a resnet model with VICReg", add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="Architecture of the backbone encoder network")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")#TODO
        parser.add_argument("--mlp", default="8192-8192-8192", help='Size and number of layers of the MLP expander head')#TODO
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # data
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default="./data/image/", help="path to download data")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=0, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        # Loss
        parser.add_argument("--invariance-coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
        parser.add_argument("--variance-coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
        parser.add_argument("--covariance-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

        return parser


def cli_main():
    # model args
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    args = parser.parse_args()

    # data module call
    dm = PreTrainDataModule(
        data_root_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
    )

    # Dataset args
    if args.dataset=="cifar10":
        # transform args
        args.input_height = 32
        args.gaussian_blur = False
        args.jitter_strength = 0.5
        normalization = cifar10_normalization()

    elif args.dataset=="imagenet":
        normalization = imagenet_normalization()
    
    # Data Augmentations
    dm.train_transforms = VICRegTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = VICRegEvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )
    
    #model = VICReg(**args.__dict__)
    print('x')

if __name__ == "__main__":
    cli_main()