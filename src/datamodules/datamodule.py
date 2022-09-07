import pytorch_lightning as pl

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR10, STL10, ImageNet

from augmentations.transforms import VICRegDataTransformPreTrain, VICRegDataTransformFineTune


    
class PreTrainDataModule(pl.LightningDataModule):
    """Pytorch lightning ImageNet DataModule."""
    def __init__(self, 
                data_root_dir: str = "../data/image/",
                dataset: str = "cifar10",
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
        """
        super().__init__()
        self.data_dir = data_root_dir+"/"+dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory       

        # Dataset Selection
        if dataset=="cifar10":
            self.dataset = CIFAR10
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            self.transform = VICRegDataTransformPreTrain(input_height=32, jitter_strength=0.5, normalize=self.normalize)

        elif dataset=="stl10":
            self.dataset = STL10
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = VICRegDataTransformPreTrain(input_height=96, jitter_strength=1, normalize=self.normalize)

        elif dataset=="imagenet":
            self.dataloader = ImageNet
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = VICRegDataTransformPreTrain(input_height=224, jitter_strength=1, normalize=self.normalize)
        
    def prepare_data(self):
        # download
        self.dataset(self.data_dir, train=True, download=True)
        self.dataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = self.dataset(self.data_dir, train=True, transform=self.transform)
            self.val = self.dataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)



class FineTuneDataModule(pl.LightningDataModule):
    """Pytorch lightning ImageNet DataModule."""
    def __init__(self, 
                data_root_dir: str = "../data/image/",
                dataset: str= 'cifar10',
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
        self.data_dir = data_root_dir+"/"+dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory       

        
        # Dataset Selection
        if dataset_name=="cifar10":
            self.dataset = CIFAR10
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            self.transform_train = VICRegDataTransformFineTune(train=True, input_height=32, jitter_strength=0.5, normalize=self.normalize)
            self.transform_val = VICRegDataTransformFineTune(train=False, input_height=32, jitter_strength=0.5, normalize=self.normalize)

        elif dataset_name=="stl10":
            self.dataset = STL10
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform_train = VICRegDataTransformFineTune(train=True, input_height=96, jitter_strength=1.0, normalize=self.normalize)
            self.transform_val = VICRegDataTransformFineTune(train=False, input_height=96, jitter_strength=1.0, normalize=self.normalize)

        elif dataset_name=="imagenet":
            self.dataset = ImageNet
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform_train = VICRegDataTransformFineTune(train=True, input_height=224, jitter_strength=1.0, normalize=self.normalize)
            self.transform_val = VICRegDataTransformFineTune(train=False, input_height=224, jitter_strength=1.0, normalize=self.normalize)

    def prepare_data(self):
        # download
        self.dataset(self.data_dir, train=True, download=True)
        self.dataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = self.dataset(self.data_dir, train=True, transform=self.transform_train)
            self.val = self.dataset(self.data_dir, train=False, transform=self.transform_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = self.dataset(self.data_dir, train=False, transform=self.transform_val)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)


