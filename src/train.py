from pytorch_lightning import  Trainer
from pytorch_lightning import loggers 

from datamodule.datamodule_vicreg import DataModule
from models.vicreg import VICReg


def main(args):

    # Data Module
    datamodule = DataModule()

    # VICREg Model

    model = VICReg()

    # Trainer
    # Weight & Bias logger
    logger_wandb=loggers.WandbLogger(save_dir="logs/") 
    trainer = Trainer(logger=logger_wandb)

    # Train model
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)

