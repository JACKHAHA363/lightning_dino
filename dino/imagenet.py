from dino.utils import GaussianBlur, Solarization
from torchvision import transforms
from dino.arrow_datasets import VisualGenomeCaptionDataset, CocoCaptionKarpathyDataset, ConceptualCaptionDataset, SBUCaptionDataset
from torch.utils.data import Dataset
from PIL import Image
import random
import os
from torchvision.datasets import ImageFolder


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

class ImageNet(Dataset):
    prompt_template = "This is a photo of {label}."
    def __init__(self, root, split, transforms, 
                 arrow_root, tokenizer, text_dataset=None, max_text_len=40, num_classes=1000):
        self.imagefolder = MyImageFolder(
            os.path.join(root, split))
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # Use only partial classes to speed up
        self.imagefolder.shrink(num_classes)
        print('Imagenet {} Total images: {}'.format(split, len(self)))

        # Read text
        caption_dset_cls = {'sbu': SBUCaptionDataset, 'coco': CocoCaptionKarpathyDataset,
                                 'cc': ConceptualCaptionDataset, 'vg': VisualGenomeCaptionDataset}.get(text_dataset, None)
        self.text_dataset = None
        if caption_dset_cls is not None:
            self.text_dataset = caption_dset_cls(split=split, data_dir=arrow_root)

    def get_random_text(self):
        random_index = random.randint(0, len(self.text_dataset.index_mapper) - 1)

        index, caption_index = self.text_dataset.index_mapper[random_index]
        text = self.text_dataset.all_texts[index][caption_index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return text, encoding

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
        if self.text_dataset is not None:
            ret['text'] = self.get_random_text()
        return ret


def dinomulticrop(config):
    multicrop = DataAugmentationDINO(
        config['image_size'],
        config['global_crops_scale'],
        config['local_crops_scale'],
        config['local_crops_number']
    )
    return [multicrop.global_transfo1, multicrop.global_transfo2] + \
        [multicrop.local_transfo for _ in range(multicrop.local_crops_number)]

