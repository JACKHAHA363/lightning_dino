import os
import argparse
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
import pytorch_lightning as pl

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
