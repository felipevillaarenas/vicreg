import numpy as np
import torchvision.transforms as transforms

from PIL import ImageOps, ImageFilter

class VICRegDataTransform:
    """Transforms for VICReg as described in the VICReg paper."""
    def __init__(self, train=True, input_height=224, jitter_strength=1.0, normalize=None):
        """Init data module.

        The default parameters can be set using the file config.datamodule.augmentations.yaml

        Args:
            train (bool): Flag used to enable transformations during fine tuning .
            input_height (int): input_height.
            jitter_strength (float): Jitter intensity,
            normalize (array[array]): Custom normalization.
        """

        self.input_height = input_height
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        self.color_jitter = transforms.ColorJitter(
            
            brightness=0.4*self.jitter_strength, 
            contrast=0.4*self.jitter_strength, 
            saturation=0.2*self.jitter_strength, 
            hue=0.1*self.jitter_strength
        )

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])


        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.input_height, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                self.final_transform   
            ]
        )

        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.input_height, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.2),
                self.final_transform   
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.finetune_transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample), self.transform_prime(sample), self.finetune_transform(sample)


class GaussianBlur(object):
    """ Implements Gaussian blur as described in the VICReg paper."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """Implements Solarization as described in the VICReg paper."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img