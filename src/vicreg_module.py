from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from pytorch_lightning import LightningModule,Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

import vicreg_resnet as resnet

class VICReg(LightningModule):
    """PyTorch Lightning implementation of VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning_

    Paper authors: Adrien Bardes, Jean Ponce and Yann LeCun.
    
    Model implemented by:
        - `Luis Felipe Villa-Arenas <https://github.com/felipevillaarenas/vicreg>`_
    .. warning:: Work in progress. This implementation is still being verified.
    
    TODOs:
        - verify on CIFAR-10
        - verify on imagenet
    
    Example::
        model = VICReg(arch="resnet34",mlp_expander="2048-2048")
        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = VICRegTrainDataTransform(32)
        dm.val_transforms = VICRegEvalDataTransform(32)
        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)
    
    Train::
        trainer = Trainer()
        trainer.fit(model)
    
    CLI command::
        # cifar10
        python vicreg_module.py 
            --accelerator gpu
            --devices 1
            --dataset cifar10
            --arch resnet34
            --mlp_expander 2048-2048

        # imagenet
        python vicreg_module.py
            --accelerator gpu
            --devices 1
            --dataset imagenet
            --data_dir /path/to/imagenet/
            --batch_size 256
            --arch resnet50
            --mlp_expander 8192-8192-8192
    
    .. _VICReg: https://arxiv.org/pdf/2105.04906.pdf
    """


    def __init__(
        self, 
        arch: str,
        mlp_expander: str,
        invariance_coeff: float = 25.0,
        variance_coeff: float = 25.0,
        covariance_coeff: float = 1.0,
        optimizer: str = "lars",
        exclude_bn_bias: bool = False,
        weight_decay: float = 1e-6,
        learning_rate: float = 0.01,
        warmup_steps:int = 1e5,
        total_steps:int = 1e50,
        **kwargs
        ):
        """
        Args:

        """
        super().__init__()
        self.save_hyperparameters()

        # Init architecture params
        self.arch = arch
        self.mlp_expander = mlp_expander
        self.num_features = int(self.mlp_expander.split("-")[-1])
        self.backbone, self.embedding_size = self.init_backbone()
        self.projector = self.init_projector()

        # Init loss params
        self.invariance_coeff = invariance_coeff
        self.variance_coeff = variance_coeff
        self.covariance_coeff = covariance_coeff

        # Init optimizer params
        self.optimizer = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        # Init scheduler params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
  
    def init_backbone(self):
        backbone, embedding_size = resnet.__dict__[self.arch](zero_init_residual=True)
        return backbone, embedding_size 

    def init_projector(self):
        mlp_spec = f"{self.embedding_size}-{self.mlp_expander}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.backbone(x)

    def shared_step(self, batch):
        # x1, x2: batches of transform views
        (x1, x2, _), _ = batch

        # y1, y2: batches of representations
        y1 = self(x1)
        y2 = self(x2)

        # z1, z2: batches of embeddings
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        # invariance Loss
        invariance_loss = F.mse_loss(z1, z2)

        # Share operation for Variance and Covariance
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        variance_loss_z1 = torch.mean(F.relu(1 - std_z1)) / 2
        variance_loss_z2 = torch.mean(F.relu(1 - std_z2)) / 2
        variance_loss = variance_loss_z1 + variance_loss_z2

        # Covariance Loss
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        covariance_loss_z1 = self.off_diagonal(cov_z1).pow_(2).sum().div(self.num_features) 
        covariance_loss_z2 = self.off_diagonal(cov_z2).pow_(2).sum().div(self.num_features)
        covariance_loss = covariance_loss_z1 +covariance_loss_z2

        # Loss function is a weighted average of the invariance, variance and covariance terms
        loss = (
            self.invariance_coeff * invariance_loss +
            self.variance_coeff * variance_loss +
            self.covariance_coeff * covariance_loss
        )
        return loss, invariance_loss, variance_loss, covariance_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def training_step(self, batch, batch_idx):
        loss, invariance_loss, variance_loss, covariance_loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_invariance_loss", invariance_loss)
        self.log("train_variance_loss", variance_loss)
        self.log("train_covariance_loss", covariance_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, invariance_loss, variance_loss, covariance_loss = self.shared_step(batch)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_invariance_loss", invariance_loss)
        self.log("val_variance_loss", variance_loss)
        self.log("val_covariance_loss", covariance_loss)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
            params = []
            excluded_params = []

            for name, param in named_params:
                if not param.requires_grad:
                    continue
                elif any(layer_name in name for layer_name in skip_list):
                    excluded_params.append(param)
                else:
                    params.append(param)

            return [
                {"params": params, "weight_decay": weight_decay},
                {
                    "params": excluded_params,
                    "weight_decay": 0.0,
                },
            ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(self.warmup_steps, self.total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], description="Pretrain a resnet model with VICReg", add_help=False)

        # model architecture params
        parser.add_argument("--arch", default="resnet50", type=str, help="Architecture of the backbone encoder network")
        parser.add_argument("--mlp_expander", default="8192-8192-8192",help='Size and number of layers of the MLP expander head')

        # data
        parser.add_argument("--dataset", default="cifar10", type=str, help="cifar10, imagenet")
        parser.add_argument("--data_dir", default="./data/image/cifar10", type=str, help="path to download data")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per device")#2048

        # transform params
        parser.add_argument("--gaussian_blur", default=True, type=bool, help="add gaussian blur")
        parser.add_argument("--jitter_strength", default=1.0, type=float, help="jitter strength")

        # Optim params
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", default=False, type=bool, help="exclude bn/bias from weight decay")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=0.01, type=float, help="base learning rate")#0.2
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--max_epochs", default=1000, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="Training will stop if max_steps or max_epochs have reached (earliest)")

        # Loss
        parser.add_argument("--invariance-coeff", default=25.0, type=float, help='invariance regularization loss coefficient')
        parser.add_argument("--variance-coeff", default=25.0, type=float, help='variance regularization loss coefficient')
        parser.add_argument("--covariance-coeff", default=1.0, type=float, help='covariance regularization loss coefficient')

        # Trainer & Infrastructure
        parser.add_argument("--accelerator", default="auto", type=str, help="supports passing different accelerator types ('cpu', 'gpu', 'tpu', 'ipu', 'auto') as well as custom accelerator instances")
        parser.add_argument("--devices", default=1, type=int, help="number of devices to train on")
        parser.add_argument("--num_workers", default=4, type=int, help="num of workers per device")
        parser.add_argument("--num_nodes", default=1, type=int, help="num of nodes")

        # Online Finetune
        parser.add_argument("--online_ft", default=False, type=bool, help="enable online evaluator")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
    from transforms import VICRegTrainDataTransform, VICRegEvalDataTransform

    # model args
    parser = ArgumentParser()
    parser = VICReg.add_model_specific_args(parser)
    args = parser.parse_args()

    # Dataset 
    if args.dataset=="cifar10":
        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        
        # Transform params defined by the dataset type
        args.input_height = dm.dims[-1]
        args.num_classes=dm.num_classes
        args.num_samples = dm.num_samples
        normalization = cifar10_normalization()
        
    elif args.dataset=="imagenet":
        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        # Transform params defined by the dataset type
        args.input_height = dm.dims[-1]
        args.num_classes =dm.num_classes
        args.num_samples = dm.num_samples
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
    
    # Distributed params
    args.world_size = args.num_nodes * args.devices if args.devices > 0 else 1
    args.train_iters_per_epoch = args.num_samples // args.world_size

    # Scheduler params
    args.warmup_steps = args.train_iters_per_epoch * args.warmup_epochs
    args.total_steps = args.train_iters_per_epoch * args.max_epochs

    model = VICReg(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=model.embedding_size,
            num_classes=args.num_classes,
            dataset=args.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if args.devices > 1 else None,
        sync_batchnorm=True if args.devices > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    cli_main()