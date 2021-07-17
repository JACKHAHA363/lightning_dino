import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import dino.vision_transformer as vits
import dino.utils as utils
from dino.eval_knn import KnnModule

class DINOModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        
        student = vits.__dict__[config['arch']](
            patch_size=config['patch_size'],
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[config['arch']](
            patch_size=config['patch_size'])
        embed_dim = student.embed_dim
   
        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = utils.MultiCropWrapper(student, vits.DINOHead(
            embed_dim, 
            config['nmb_centroids'],
            use_bn=config['use_bn_in_head'],
            norm_last_layer=config['norm_last_layer']))
        self.teacher = utils.MultiCropWrapper(
            teacher,
            vits.DINOHead(
                embed_dim, config['nmb_centroids'], 
                config['use_bn_in_head']))

        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {config['arch']} network.")

        # ============ preparing loss ... ============
        self.dino_loss = utils.DINOLoss(
            config['nmb_centroids'],
            config['local_crops_number'] + 2,  # total number of crops = 2 global crops + local_crops_number
            config['warmup_teacher_temp'],
            config['teacher_temp'],
            config['warmup_teacher_temp_epoch'],
            config['max_epoch'])

        # for online KNN accuracy
        global_batch_size = utils.get_world_size() * config['per_gpu_batchsize']
        self.num_negatives = global_batch_size * 100
        self.num_classes = 1000
        self.register_buffer("queue", torch.randn(embed_dim, self.num_negatives))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels", torch.zeros(self.num_negatives))

    def _enqueue(self, features, labels):
        ptr = int(self.queue_ptr)
        batch_size = features.shape[0]
        final_batch_size = self.queue[:, ptr: ptr + batch_size].shape[1]

        # replace the keys at ptr (dequeue and enqueue)
        labels = labels[:final_batch_size]
        features = F.normalize(features[:final_batch_size], dim=1, p=2)
        self.queue[:, ptr:ptr + batch_size] = features.T
        self.queue_labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.num_negatives  # move pointer
        self.queue_ptr[0] = ptr

    def compute_dino(self, batch):
        """ The input is image only """
        it = self.trainer.global_step
        epoch = self.trainer.current_epoch
        optimizer = self.trainer.optimizers[0]
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]
            self.log(f"lr-AdamW/pg{i+1}", param_group['lr'])
        
        images = batch[self.config['dino_img_key']]
        teacher_output, teacher_embs = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output, student_embs = self.student(images)
        loss, temp = self.dino_loss(student_output, teacher_output, epoch)
        
        self.log(f"dino/teacher_temperature", temp)
        phase = "train" if self.training else "val"
        self.log(f"dino/{phase}/loss", loss)
        phase = "train" if self.training else "val"
        self.log("dino/ema_momentum", self.momentum_schedule[it])
        self.log('dino/weight_decay', self.wd_schedule[it])

        # compute variance
        with torch.no_grad():
            last_layer = self.student.head.last_layer
            self.log("dino/last_layer_norm_avg", last_layer.weight_g.mean().item())
            self.log("dino/last_layer_direction_var", last_layer.weight_v.var(dim=0).mean().item())
            self.log("dino/center_mean", self.dino_loss.center.mean().item())
            self.log("dino/center_var", self.dino_loss.center.var().item())
            self.log("dino/teacher_var", teacher_embs.var(dim=0).mean().item())
            self.log("dino/student_var", student_embs.var(dim=0).mean().item())
        
        return {'loss': loss, 
                'features': teacher_embs[:images[0].size(0), :].detach(),
                'labels': torch.tensor(batch[self.config['dino_label_key']]).long().cuda()}
    
    def training_step(self, batch, batch_idx):
        output = self.compute_dino(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        # enqueue only one view
        self._enqueue(output['features'], output['labels'])
  
        return total_loss

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        loss.backward()
        epoch = self.trainer.current_epoch
        utils.cancel_gradients_last_layer(epoch, self.student, self.config['freeze_last_layer'])

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int, optimizer_closure, on_tpu: bool, using_native_amp: bool, using_lbfgs: bool) -> None:
        super().optimizer_step(epoch=epoch, batch_idx=batch_idx, optimizer=optimizer, optimizer_idx=optimizer_idx, optimizer_closure=optimizer_closure, on_tpu=on_tpu, using_native_amp=using_native_amp, using_lbfgs=using_lbfgs)
        
        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[self.trainer.global_step]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
 
    def validation_step(self, batch, batch_idx):
        output = self.compute_dino(batch)
        test_labels = output['labels']
        test_features = output['features']
        test_features = F.normalize(test_features, dim=1, p=2)
        top1, top5 = KnnModule.do_knn_step(self.queue, self.queue_labels, test_features, test_labels,
                                           T=0.07, k=20, num_classes=self.num_classes)
        self.log('dino/knn_top1', top1, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('dino/knn_top5', top5, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(self.student)
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        
        # ============ init schedulers ... ============
        data_loader = self.trainer.datamodule.train_dataloader()
        config = self.config
        self.lr_schedule = utils.cosine_scheduler(
            config['learning_rate'] * (config['per_gpu_batchsize'] * utils.get_world_size()) / 256.,  # linear scaling rule
            config['end_lr'],
            config['max_epoch'], len(data_loader),
            warmup_epochs=config['warmup_epoch'])
        self.wd_schedule = utils.cosine_scheduler(
            config['weight_decay'],
            config['weight_decay_end'],
            config['max_epoch'], len(data_loader))
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(
            config['momentum_teacher'], 1,
            config['max_epoch'], len(data_loader))
        print(f"Loss, optimizer and schedulers ready.")
        return optimizer
