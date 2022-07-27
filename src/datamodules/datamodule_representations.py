import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import CIFAR10, ImageNet
from augmentations.transforms import VICRegDataTransformTrain


class CIFAR10DataModule(pl.LightningDataModule):
    """Pytorch lightning CIFAR10 DataModule."""
    def __init__(self, data_dir: str = "../data/image/cifar10/"):
        """Init data module.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            data:dir (string): Data path directory.
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = VICRegDataTransformTrain(input_height=224, jitter_strength=1.0, normalize=None)

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.val = CIFAR10(self.data_dir, train=False, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR10(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)



