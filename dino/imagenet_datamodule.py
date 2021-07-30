from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
from pytorch_lightning import LightningDataModule
from dino.imagenet import ImageNet, dinomulticrop
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import functools
import torchvision.transforms as pth_transforms


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

def normal(config):
    return [pth_transforms.Compose([
                       pth_transforms.Resize(256, interpolation=3),
                       pth_transforms.CenterCrop(224),
                       pth_transforms.ToTensor(),
                       pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])]

def key_to_transforms(key):
    return {'multicrop': dinomulticrop,
            'normal': normal}[key]

class ImageNetDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.config = _config
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.root =  _config.get('imagenet_dir', None)
        
        self.train_transforms = key_to_transforms(_config['train_transform'])(_config)
        self.val_transforms = key_to_transforms(_config['val_transform'])(_config)
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

        self.train_dataset = ImageNet(
            root=self.root, split='train',
            transforms=self.train_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])

        self.val_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])

        self.test_dataset = ImageNet(
            root=self.root, split='val', 
            transforms=self.val_transforms,
            arrow_root=self.config['data_root'],
            tokenizer=self.tokenizer,
            text_dataset=self.config['text_dataset'],
            max_text_len=self.config['max_text_len'])
        self.num_samples = len(self.train_dataset)

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
