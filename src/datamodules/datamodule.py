import pytorch_lightning as pl

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from augmentations.transforms import VICRegDataTransformPreTrain, VICRegDataTransformFineTune


    
class ImageNetPreTrainDataModule(pl.LightningDataModule):
    """Pytorch lightning ImageNet DataModule."""
    def __init__(self, 
                data_dir: str = "../data/image/imagenet/",
                batch_size: int = 32,
                num_workers: int = 2, 
                pin_memory: bool = False,
                finetune: bool = False
                ):
        """Init data module.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            data:dir (string): Data path directory.
            batch_size (int): Number of samples per batch to load.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            finetune (bool): Selects data transformation for finetune the model for Evaluation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory       

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = VICRegDataTransformPreTrain(input_height=224, jitter_strength=1.0, normalize=self.normalize)

    def prepare_data(self):
        # download
        ImageNet(self.data_dir, train=True, download=True)
        ImageNet(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = ImageNet(self.data_dir, train=True, transform=self.transform)
            self.val = ImageNet(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)



class ImageNetFineTuneDataModule(pl.LightningDataModule):
    """Pytorch lightning ImageNet DataModule."""
    def __init__(self, 
                data_dir: str = "../data/image/imagenet/",
                batch_size: int = 32,
                num_workers: int = 2, 
                pin_memory: bool = False,
                ):
        """Init data module.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            data:dir (string): Data path directory.
            batch_size (int): Number of samples per batch to load.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            finetune (bool): Selects data transformation for finetune the model for Evaluation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory       

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_train = VICRegDataTransformFineTune(train=True, input_height=224, jitter_strength=1.0, normalize=self.normalize)
        self.transform_val = VICRegDataTransformFineTune(train=False, input_height=224, jitter_strength=1.0, normalize=self.normalize)

    def prepare_data(self):
        # download
        ImageNet(self.data_dir, train=True, download=True)
        ImageNet(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = ImageNet(self.data_dir, train=True, transform=self.transform_train)
            self.val = ImageNet(self.data_dir, train=False, transform=self.transform_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = ImageNet(self.data_dir, train=False, transform=self.transform_val)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)


