import numpy as np
import torchvision.transforms as transforms

from PIL import ImageOps, ImageFilter

class VICRegTrainDataTransform:
    """Transforms for VICReg implemented as in paper's author repository.
    """
    def __init__(
        self, 
        input_height: int = 224, 
        gaussian_blur: bool = True,
        jitter_strength: float = 1, 
        normalize=None
        ):
        """
        Args:
            input_height (int): input_height.
            jitter_strength (float): Jitter intensity.
            gaussian_blur (bool): Enable gaussian blur transform.
            normalize (array[array]): Custom normalization.
        """
        super().__init__()

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

        self.prime_transform = transforms.Compose(
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

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose(
            [transforms.RandomResizedCrop(self.input_height), transforms.RandomHorizontalFlip(), self.final_transform]
        )

    def __call__(self, sample):
        return self.transform(sample), self.prime_transform(sample), self.online_transform(sample)


class VICRegEvalDataTransform(VICRegTrainDataTransform):
    """ Transforms for VICReg implemented as in paper's author repository."""
    def __init__(
        self, 
        input_height: int = 224, 
        gaussian_blur: bool = True,
        jitter_strength: float = 1, 
        normalize=None
        ):
        """
        Args:
            input_height (int): input_height.
            jitter_strength (float): Jitter intensity.
            gaussian_blur (bool): Enable gaussian blur transform.
            normalize (array[array]): Custom normalization.
        """
        super().__init__(normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength)

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class GaussianBlur(object):
    """Gaussian blur implemented as in paper's author repository."""
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
    """Solarization implemented as in paper's author repository."""
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img