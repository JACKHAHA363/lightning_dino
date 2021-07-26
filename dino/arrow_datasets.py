from torch.utils.data import Dataset
import os

import pyarrow as pa
import torch
import random

class ArrowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        tokenizer=None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_column_name = text_column_name
        self.names = names
        self.data_dir = data_dir
        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if not isinstance(self.all_texts[0][0], str):
                    for texts in self.all_texts:
                        for idx, text in enumerate(texts):
                            texts[idx] = text.item()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "":
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return self.dataset_length


class ConceptualCaptionDataset(ArrowDataset):
    def __init__(self, split, data_dir):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"conceptual_caption_train_{i}" for i in range(31)]
        elif split == "val":
            names = ["conceptual_caption_val_0"]

        super().__init__(data_dir=data_dir, names=names, text_column_name="caption")


class CocoCaptionKarpathyDataset(ArrowDataset):
    def __init__(self, split, data_dir):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ["coco_caption_karpathy_test"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]

        super().__init__(data_dir=data_dir, names=names, text_column_name="caption")


class SBUCaptionDataset(ArrowDataset):
    def __init__(self, split, data_dir):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"sbu_{i}" for i in range(10)]
        elif split == "val":
            names = []

        super().__init__(data_dir=data_dir, names=names, text_column_name="caption")


class VisualGenomeCaptionDataset(ArrowDataset):
    def __init__(self, split, data_dir):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = ["vg"]
        elif split == "val":
            names = []

        super().__init__(data_dir=data_dir, names=names, text_column_name="caption")

