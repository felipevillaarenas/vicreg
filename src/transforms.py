import numpy as np

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg


if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _PIL_AVAILABLE:
    from PIL import ImageOps, ImageFilter
else:  # pragma: no cover
    warn_missing_pkg("PIL")



class VICRegTrainDataTransform:
    """Transforms for VICReg.

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from src.transforms import SimCLRTrainDataTransform

        transform = VICRegTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, 
        input_height: int = 224, 
        gaussian_blur: bool = True, 
        jitter_strength: float = 1.0, 
        normalize=None
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.2, 
            hue=0.1
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
        ]

        data_transforms_prime = [
            transforms.RandomResizedCrop(size=self.input_height, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.2),
        ]

        data_transforms = transforms.Compose(data_transforms)
        data_transforms_prime = transforms.Compose(data_transforms_prime)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])
        self.train_transform_prime = transforms.Compose([data_transforms_prime, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_height), 
                transforms.RandomHorizontalFlip(), 
                self.final_transform
            ]
        )

    def __call__(self, sample):
        transform = self.train_transform
        transform_prime = self.train_transform_prime

        x = transform(sample)
        x_prime = transform_prime(sample)

        return x, x_prime, self.online_transform(sample)


class VICRegEvalDataTransform(VICRegTrainDataTransform):
    """Transforms for VICReg.

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from src.transforms import VICRegEvalDataTransform

        transform = VICRegEvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ):
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class VICRegFinetuneTransform:
    def __init__(
        self, 
        input_height: int = 224, 
        jitter_strength: float = 1.0, 
        normalize=None, 
        eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.2, 
            hue=0.1
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the VICReg paper

    def __init__(self, p):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `ImageFilter` from `PIL` which is not installed yet.")

        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    # Implements Solarization as described in the VICReg paper
    def __init__(self, p):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `ImageOps` from `PIL` which is not installed yet.")

        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img