import pytorch_lightning as pl
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision import transforms
from dino.utils import GaussianBlur, Solarization
from PIL import Image

def dinomulticrop(config):
    multicrop = DataAugmentationDINO(
        config['image_size'],
        config['global_crops_scale'],
        config['local_crops_scale'],
        config['local_crops_number']
    )
    return [multicrop.global_transfo1, multicrop.global_transfo2] + \
        [multicrop.local_transfo for _ in range(multicrop.local_crops_number)]

class DataAugmentationDINO(object):
    def __init__(self, size, global_crops_scale, local_crops_scale, local_crops_number):
        global_size = size
        local_size = int(size * 96 / 224)
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])


class MyImageFolder(ImageFolder):
    def shrink(self, num_classes):
        """ Shrink to desired number of classes results """
        self.samples = [self.samples[i] for i in range(len(self.samples)) if self.samples[i][1] < num_classes]

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, num_classes=1000):
        self.imagefolder = MyImageFolder(
            os.path.join(root, split))
        self.transforms = transforms
     
        # Use only partial classes to speed up
        self.imagefolder.shrink(num_classes)
        print('Imagenet {} Total images: {}'.format(split, len(self)))

    def set_transforms(self, transforms):
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

    def __len__(self):
        return len(self.imagefolder)
    
    def __getitem__(self, item):
        images, label = self.imagefolder[item]
        ret = {}
        ret['imagenet_image'] = [tr(images) for tr in self.transforms]
        ret['imagenet_label'] = label
        return ret

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.root = _config.get('imagenet_dir', None)
        self.train_transforms = dinomulticrop(_config)
        self.val_transforms = dinomulticrop(_config)
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

    def setup(self, stage=None):
        self.train_dataset = ImageNet(
            root=self.root, split='train',
            transforms=self.train_transforms)

        self.val_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms)

        self.test_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader
