import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR10, ImageNet

    
class PreTrainDataModule(pl.LightningDataModule):
    """Pytorch lightning ImageNet DataModule."""
    def __init__(self, 
                data_root_dir: str = "../data/image/",
                dataset: str = "cifar10",
                batch_size: int = 32,
                num_workers: int = 2, 
                pin_memory: bool = True
                ):
        """
        Args:
            data:dir (str): Data path directory.
            dataset (str): Dataset name
            batch_size (int): Number of samples per batch to load.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__()
        self.data_dir = data_root_dir+"/"+dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        # Dataset Selection
        if dataset=="cifar10":
            self.dataset = CIFAR10

        elif dataset=="imagenet":
            self.dataloader = ImageNet

        
    def prepare_data(self):
        # download
        self.dataset(self.data_dir, train=True, download=True)
        self.dataset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(self.data_dir, train=True, transform=self.train_transforms)
            self.val_dataset = self.dataset(self.data_dir, train=False, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)
