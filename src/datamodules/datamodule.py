import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import CIFAR10, ImageNet
from augmentations.transforms import VICRegDataTransformTrain, VICRegDataTransformEval


class CIFAR10DataModule(pl.LightningDataModule):
    """Pytorch lightning CIFAR10 DataModule."""
    def __init__(self, 
                 data_dir: str = "../data/image/cifar10/",
                 batch_size: int = 32,
                 num_workers: int = 8, 
                 pin_memory: bool = False,
                 finetune: bool = False
                 ):
        """Init data module.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            data:dir (string): Data path directory.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory       
        self.finetune = finetune

        if self.finetune:
            self.transform_train = VICRegDataTransformEval(finetune=True, input_height=224, jitter_strength=1.0, normalize=None)
            self.transform_val = VICRegDataTransformEval(finetune=False, input_height=224, jitter_strength=1.0, normalize=None)
        else:
            self.transform_train = VICRegDataTransformTrain(input_height=224, jitter_strength=1.0, normalize=None)
            self.transform_val = VICRegDataTransformTrain(input_height=224, jitter_strength=1.0, normalize=None)

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.val = CIFAR10(self.data_dir, train=False, transform=self.transform_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = CIFAR10(self.data_dir, train=False, transform=self.transform_val)


    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)



