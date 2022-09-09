import numpy as np
import torchvision.transforms as transforms

from PIL import ImageOps, ImageFilter

class VICRegTrainDataTransform:
    """Transforms for VICReg as described in the VICReg paper."""
    def __init__(
        self, 
        input_height: int, 
        gaussian_blur: bool,
        jitter_strength: float, 
        normalize=None
        ):
        """Init Class VICRegDataTransform.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            input_height (int): input_height.
            jitter_strength (float): Jitter intensity,
            normalize (array[array]): Custom normalization.
        """

        self.input_height = input_height
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.gaussian_blur = gaussian_blur

        self.color_jitter = transforms.ColorJitter(
            
            brightness=0.4*self.jitter_strength, 
            contrast=0.4*self.jitter_strength, 
            saturation=0.2*self.jitter_strength, 
            hue=0.1*self.jitter_strength
        )

        # Enable normalization
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
                GaussianBlur(p=1.0, active=self.gaussian_blur),
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
                GaussianBlur(p=1.0, active=self.gaussian_blur),
                Solarization(p=0.2),
                self.final_transform   
            ]
        )

    def __call__(self, sample):
        return self.transform(sample), self.transform_prime(sample)

class VICRegEvalDataTransform:
    """Transforms for VICReg as described in the VICReg paper for Eval."""
    def __init__(
        self, 
        input_height: int, 
        gaussian_blur: bool,
        jitter_strength: float, 
        normalize=None
        ):
        """Init Class VICRegDataTransform.

        The default parameters can be set using the file config.datamodules.augmentations.yaml

        Args:
            train (bool): Define if the transformation is used un the train dataloader
            input_height (int): input_height.
            normalize (array[array]): Custom normalization.
        """
        self.input_height = input_height
        self.normalize = normalize

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(),
                self.final_transform   
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur(object):
    """ Implements Gaussian blur as described in the VICReg paper."""
    def __init__(self, p, active):
        self.p = p
        self.active = active

    def __call__(self, img):
        if np.random.rand() < self.p and self.active:
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