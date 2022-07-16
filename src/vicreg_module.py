import math
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor, nn
from torch.nn import functional as F

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

class VICReg(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    
    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)








    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
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

        return parser

    def cli_main():
        from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
        from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
        from transforms import VICRegEvalDataTransform, VICRegTrainDataTransform

        parser = ArgumentParser()

        # model args
        parser = VICReg.add_model_specific_args(parser)
        args = parser.parse_args()

        if args.dataset == "stl10":
            
            # Update args
            args.maxpool1 = False
            args.first_conv = True
            args.gaussian_blur = True
            args.jitter_strength = 1.0

            # Normalization
            normalization = stl10_normalization()

            # Datamodule
            dm = STL10DataModule(
                data_dir=args.data_dir, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers
            )

            dm.train_dataloader = dm.train_dataloader_mixed
            dm.val_dataloader = dm.val_dataloader_mixed

            # Update args from Datamodule
            args.input_height = dm.size()[-1]
            args.num_samples = dm.num_unlabeled_samples

        elif args.dataset == "cifar10":

            # Update args
            args.maxpool1 = False
            args.first_conv = False
            args.temperature = 0.5
            args.gaussian_blur = False
            args.jitter_strength = 0.5

            # Normalization
            normalization = cifar10_normalization()
            
            # Define validation split
            val_split = 5000
            if args.num_nodes * args.gpus * args.batch_size > val_split:
                val_split = args.num_nodes * args.gpus * args.batch_size

            # Datamodule
            dm = CIFAR10DataModule(
                data_dir=args.data_dir, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, 
                val_split=val_split
            )

            # Update args from Datamodule
            args.num_samples = dm.num_samples
            args.input_height = dm.size()[-1]
        

        elif args.dataset == "imagenet":
            
            # Update args
            args.maxpool1 = True
            args.first_conv = True
            args.gaussian_blur = True
            args.jitter_strength = 1.0
            args.batch_size = 64
            args.num_nodes = 8
            args.gpus = 8  # per-node
            args.max_epochs = 800
            args.optimizer = "lars"
            args.learning_rate = 4.8
            args.final_lr = 0.0048
            args.start_lr = 0.3
            args.online_ft = True
            
            # Normalization
            normalization = imagenet_normalization()

            # Datamodule
            dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

            # Update args from Datamodule
            args.num_samples = dm.num_samples
            args.input_height = dm.size()[-1]

            
        else:
            raise NotImplementedError("other datasets have not been implemented till now")

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
                num_classes=dm.num_classes,
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