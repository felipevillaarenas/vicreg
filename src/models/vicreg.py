import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import AdamW

class VICReg(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = self.init_projector()


    def forward(self, x, y):
        return self.projector(self.backbone(x))

    def init_projector(self):
        mlp_spec = f"{self.embedding}-{self.args.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)  
    
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def shared_step(self, batch, batch_idx):
        x, y = batch
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        # invariance Loss
        invariance_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0) # TODO
        y = torch.cat(FullGatherLayer.apply(y), dim=0) # TODO

        # Variance Loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        variance_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Covariance Loss
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        covariance_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.invariance_coeff * invariance_loss
            + self.args.variance_coeff * variance_loss
            + self.args.covariance_coeff * covariance_loss
        )
        return loss, invariance_loss, variance_loss, covariance_loss

    def training_step(self, batch, batch_idx):
        loss, invariance_loss, variance_loss, covariance_loss = self.shared_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/invariance_loss", invariance_loss)
        self.log("train/variance_loss", variance_loss)
        self.log("train/covariance_loss", covariance_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, invariance_loss, variance_loss, covariance_loss = self.shared_step(batch)
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/invariance_loss", invariance_loss)
        self.log("val/variance_loss", variance_loss)
        self.log("val/covariance_loss", covariance_loss)
        return loss
  

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
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
