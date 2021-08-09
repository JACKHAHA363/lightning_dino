import os
import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms as pth_transforms
import pytorch_lightning as pl

from dino.config import ex
from dino.modules import DINOModel
from dino.imagenet_datamodule import ImageNetDataModule
import tqdm
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--ckpt')
args = parser.parse_args()
model = DINOModel.load_from_checkpoint(args.ckpt)
model.cuda()

_config = model.config.copy()
_config['train_transform'] = 'normal'
_config['val_transform'] = 'normal'
dm = ImageNetDataModule(_config)
tokenizer = dm.tokenizer

model.eval()
val_loader = dm.val_dataloader()
conf_mat = torch.zeros(1000, tokenizer.vocab_size).cuda()
for batch in tqdm.tqdm(val_loader):
    with torch.no_grad():
        logits, _ = model.teacher(batch['imagenet_image'][0].cuda())
        probs = nn.functional.softmax(logits, dim=-1)
    for prob, label in zip(probs, batch['imagenet_label']):
        conf_mat[label.cuda()] += prob
torch.save(
    {'conf_mat': conf_mat, 
     'classes': val_loader.dataset.imagefolder.classes}, 
    "./conf_mat.pkl")