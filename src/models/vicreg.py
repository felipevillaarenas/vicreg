

import torch
import torch.nn.functional as F

from torch import nn
from pytorch_lightning import LightningModule

from components import resnet


class VICReg(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = self.Projector(args, self.embedding)

        self.save_hyperparameters()

        # loss function
        self.criterion_mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return self.projector(self.backbone(x))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)

    def Projector(self, args, embedding):
        mlp_spec = f"{embedding}-{args.mlp}"
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
    
    def step(self, batch):
        input, labels = batch
        x, y = input
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        # Invariance Loss
        inv_loss = F.mse_loss(x, y)

        # Variance and Covariance Loss
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        var_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        # Combine losses
        loss = (
            self.args.sim_coeff * inv_loss
            + self.args.std_coeff * var_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss , inv_loss, var_loss, cov_loss

    
    def training_step(self, batch, batch_idx):
        loss , inv_loss, var_loss, cov_loss= self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/invariance_loss", inv_loss)
        self.log("train/variance_loss", var_loss)
        self.log("train/covariance_loss", cov_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss , inv_loss, var_loss, cov_loss = self.step(batch)
        return {"loss": loss, "invariance_loss": inv_loss, "variance_loss": var_loss, "covariance_loss": cov_loss}



       


