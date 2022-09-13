import os
from pathlib import Path
from argparse import ArgumentParser


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import pytorch_lightning as pl

from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

import backbone as resnet


class VICReg(pl.LightningModule):
    """PyTorch Lightning implementation of VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning_

    Paper authors: Adrien Bardes, Jean Ponce and Yann LeCun.
    
    Model implemented by:
        - `Luis Felipe Villa-Arenas <https://github.com/felipevillaarenas/vicreg>`_
    .. warning:: Work in progress. This implementation is still being verified.
    
    TODOs:
        - verify on CIFAR-10
        - verify on imagenet
    
    Example::
        model = VICReg(num_classes=10)
        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)
    
    Train::
        trainer = Trainer()
        trainer.fit(model)
    
    CLI command::
        # cifar10
        python vicreg_module.py --gpus 1
        # imagenet
        python vicreg_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    
    .. _VICReg: https://arxiv.org/pdf/2105.04906.pdf
    """


    def __init__(
        self, 
        arch: str,
        mlp_expander: str,
        
        invariance_coeff: float,
        variance_coeff: float,
        covariance_coeff: float,
        optimizer: str,
        exclude_bn_bias: bool,
        weight_decay: float,
        learning_rate: float,
        warmup_epochs: int,
        max_epochs: int,
        batch_size: int,
        devices: str,
        num_nodes: int,
        num_samples:int,

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
        self.num_features = int(self.mlp.split("-")[-1])
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
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size  = batch_size

        # Init infrastructure
        self.devices = devices
        self.num_nodes = num_nodes

        # Data specific
        self.num_samples = num_samples

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.devices * self.batch_size if self.devices > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size



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
        cov_z1 = (z1.T @ z1) / (self.batch_size - 1)
        cov_z2 = (z2.T @ z2) / (self.batch_size - 1)
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

    def training_step(self, batch):
        loss, invariance_loss, variance_loss, covariance_loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_invariance_loss", invariance_loss)
        self.log("train_variance_loss", variance_loss)
        self.log("train_covariance_loss", covariance_loss)
        return loss

    def validation_step(self, batch):
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

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    
    