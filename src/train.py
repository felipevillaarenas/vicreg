
import os
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import distributed as dist
from torch import nn

from vicreg import VICReg

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
)


def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], description="Pretrain a resnet model with VICReg", add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="Architecture of the backbone encoder network")
        parser.add_argument("--mlp", default="8192-8192-8192",help='Size and number of layers of the MLP expander head')

        # data
        parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, imagenet")
        parser.add_argument("--data_dir", type=str, default="./data/image/", help="path to download data")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=0, type=int, help="num of workers per GPU")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        # Optim params
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

        # Loss
        parser.add_argument("--invariance-coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
        parser.add_argument("--variance-coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
        parser.add_argument("--covariance-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

        # Online Finetune
        parser.add_argument("--online_ft", action="store_true", help="Enable online evaluator")


        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
    from transforms import VICRegTrainDataTransform, VICRegEvalDataTransform

    # model args
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    args = parser.parse_args()


    # Dataset 
    if args.dataset=="cifar10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        
        # Transform params defined by the dataset type
        args.input_height = dm.dims[-1]
        args.num_classes=dm.num_classes
        normalization = cifar10_normalization()
        

    elif args.dataset=="imagenet":
        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        # Transform params defined by the dataset type
        args.input_height = dm.dims[-1]
        args.num_classes =dm.num_classes
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
    
    model = VICReg(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=args.num_classes,
            dataset=args.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    print('x')

if __name__ == "__main__":
    cli_main()