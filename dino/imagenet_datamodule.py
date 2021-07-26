from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import os
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from transformers.models.bert import tokenization_bert
from dino.utils import GaussianBlur, Solarization
from PIL import Image
import torch
from dino.arrow_datasets import VisualGenomeCaptionDataset, CocoCaptionKarpathyDataset, ConceptualCaptionDataset, SBUCaptionDataset
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import functools
import random


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )

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

        
class ImageNetDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.config = _config
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.root =  _config.get('imagenet_dir', None)
        
        self.train_transforms = dinomulticrop(_config)
        self.val_transforms = dinomulticrop(_config)
        self.setup_flag = False
        
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.collate_fn = functools.partial(
            collate, mlm_collator=self.mlm_collator,
        )

    def set_train_dataset(self):
        self.train_dataset = ImageNet(
            root=self.root, split='train',
            transforms=self.train_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])

    def set_val_dataset(self):
        self.val_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])

    def set_test_dataset(self):
        self.test_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])

    def setup(self, stage):
        self.set_train_dataset()
        self.set_val_dataset()
        self.set_test_dataset()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        return loader


def collate(batch, mlm_collator):
    keys = set([key for b in batch for key in b.keys()])

    # Non text
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    for key in dict_batch:
        if 'text' not in key:
            dict_batch[key] = default_collate(dict_batch[key])

    # Processed text
    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        batch_size = len(texts[0])
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask
    return dict_batch
