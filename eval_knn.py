import os
import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms as pth_transforms
import pytorch_lightning as pl

from dino.config import ex
from dino.modules import DINOModel
from dino.imagenet_datamodule import ImageNetDataModule
import copy

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class KnnModule(pl.LightningModule):
    def __init__(self, K, t, ckpt_path, num_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.K = K
        self.t = t
        self.features = None
        self.labels = None
        # pretrained model
        self.num_classes = num_classes
        self.model = DINOModel.load_from_checkpoint(ckpt_path)
    
    @property
    def automatic_optimization(self) -> bool:
        return False

    @staticmethod
    def do_knn_step(training_features, training_labels, test_features, test_labels, T, k, num_classes):
        retrieval_one_hot = torch.zeros(k, num_classes).to(test_features.device)

        # calculate the dot product and compute top-k neighbors
        batch_size = test_features.size(0)
        similarity = torch.mm(test_features, training_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = training_labels.view(1, -1).expand(batch_size, -1).long()
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ), 1)
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(test_labels.data.view(-1, 1))
        top1 = correct.narrow(1, 0, 1).sum()
        top5 = correct.narrow(1, 0, 5).sum()
        return top1, top5

    def forward(self, image):
        with torch.no_grad():
            features = self.model.teacher(image)[-1]
            features = nn.functional.normalize(features, dim=1, p=2)
        return features

    def _enqueue(self, features, labels):
        features = features
        labels = labels
        if self.features is None:
            self.features = features.T
            self.labels = labels
        else:
            # dim x num_train_examples
            self.features = torch.cat((self.features, features.T), 1)
            self.labels = torch.cat((self.labels, labels), 0)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        features = self.forward(batch['imagenet_image'])
        labels = batch['imagenet_label']

        features = concat_all_gather(features)
        labels = concat_all_gather(labels)

        if dist.get_rank() == 0:
            self._enqueue(features, labels)
        return None

    def validation_step(self, batch, batch_idx):
        features = self.forward(batch['imagenet_image'])
        labels = batch['imagenet_label']

        features = concat_all_gather(features)
        labels = concat_all_gather(labels)

        top1 = torch.zeros(1).to(features.device)
        top5 = torch.zeros(1).to(features.device)
        if dist.get_rank() == 0 and self.features is not None:
            top1, top5 = self.do_knn_step(self.features, self.labels, features, labels,
                                          self.t, self.K, num_classes=self.num_classes)
        dist.barrier()
        dist.broadcast(top1, 0)
        dist.broadcast(top5, 0)
        return {'top1': top1, 'top5': top5, 'num_examples': features.size(0)}

    def validation_epoch_end(self, outputs):
        total = 0.
        top1, top5 = 0., 0.
        for output in outputs:
            total += output['num_examples']
            top1 += output['top1']
            top5 += output['top5']
        top1 = top1 * 100. / total
        top5 = top5 * 100. / total
        dist.barrier()
        dist.broadcast(top1, 0)
        dist.broadcast(top5, 0)

        results = {'val/top1': top1.item(), 'val/top5': top5.item()}
        self.log_dict(results, sync_dist=True)

        if dist.get_rank() == 0:
            print('Validation results:')
            print(results)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    dm = ImageNetDataModule(_config)
    knn = KnnModule(
        K=_config['nb_knn'],
        t=_config['knn_temp'],
        ckpt_path=_config['load_path'],
        num_classes=1000)
    csv_logger = pl.loggers.csv_logs.CSVLogger(_config['log_dir'], _config['exp_name'])
    trainer = pl.Trainer(
        logger=[csv_logger],
        gpus=_config['num_gpus'], 
        num_nodes=_config["num_nodes"],
        max_epochs=1, 
        num_sanity_val_steps=0,
        accelerator='ddp',
        fast_dev_run=_config['fast_dev_run'])
    trainer.fit(knn, dm)
