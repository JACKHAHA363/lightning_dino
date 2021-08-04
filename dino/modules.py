import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import dino.vision_transformer as vits
import dino.utils as utils
from dino.knn import do_knn_step
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel
from sklearn.metrics import normalized_mutual_info_score

class DINOModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config        
        student = vits.__dict__[config['arch']](
            patch_size=config['patch_size'],
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[config['arch']](
            patch_size=config['patch_size'])
        embed_dim = student.embed_dim

        # Init from pretrained word emb
        pretrained_word_embs = None
        if config['init_word_emb']:
            print('Initialize with pretrained word emb...')
            assert config['nmb_centroids'] == config['vocab_size']
            pretrained_transformer = BertModel.from_pretrained(config['tokenizer'])
            pretrained_word_embs = pretrained_transformer.embeddings.word_embeddings.weight
            assert config['vocab_size'] == pretrained_word_embs.shape[0]
            assert config['bottleneck_dim'] == pretrained_word_embs.shape[1]

        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = utils.MultiCropWrapper(student, vits.DINOHead(
            embed_dim, 
            config['nmb_centroids'],
            use_bn=config['use_bn_in_head'],
            norm_last_layer=config['norm_last_layer'],
            bottleneck_dim=config['bottleneck_dim'],
            last_layer_weight=pretrained_word_embs,
            ))
        self.teacher = utils.MultiCropWrapper(
            teacher,
            vits.DINOHead(
                embed_dim, config['nmb_centroids'], 
                config['use_bn_in_head'],
                bottleneck_dim=config['bottleneck_dim']))

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

        # MLM
        self.use_mlm = config['text_dataset'] is not None
        if self.use_mlm:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
            self.text_embeddings = BertEmbeddings(bert_config)
            self.text_embeddings.apply(vits.init_weights)
            self.mlm_head = vits.DINOMLMHead(
                embed_dim, 
                config['vocab_size'],
                norm_last_layer=config['norm_last_layer'],
                last_layer=self.student.head.last_layer,
                bottleneck_dim=config['bottleneck_dim']
                )

        # Schedules
        # ============ init schedulers ... ============
        global_batch_size = config['num_gpus'] * config['num_nodes'] * config['per_gpu_batchsize']
        print('########## global_batch_size', global_batch_size)
        train_iters_per_epoch = config['num_samples'] // global_batch_size
        self.lr_schedule = utils.cosine_scheduler(
            config['learning_rate'] * global_batch_size / 256.,  # linear scaling rule
            config['end_lr'],
            config['max_epoch'], train_iters_per_epoch,
            warmup_epochs=config['warmup_epoch'])
        self.wd_schedule = utils.cosine_scheduler(
            config['weight_decay'],
            config['weight_decay_end'],
            config['max_epoch'], train_iters_per_epoch)
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(
            config['momentum_teacher'], 1,
            config['max_epoch'], train_iters_per_epoch)
        print(f"Loss, optimizer and schedulers ready.")

    def on_train_epoch_start(self) -> None:
        # Init local memory for online KNN Eval
        self.knn_feats = []
        self.knn_labels = []

    def on_validation_epoch_start(self):
        # For online KNN
        self.knn_feats = torch.cat(self.knn_feats)
        self.knn_labels = torch.cat(self.knn_labels)

        # For NMI
        self.val_labels = []
        self.val_cluster_assignments = []

    def on_validation_epoch_end(self):
        self.val_labels = torch.cat(self.val_labels).cpu()
        self.val_cluster_assignments = torch.cat(self.val_cluster_assignments).cpu()
        nmi = normalized_mutual_info_score(labels_true=self.val_labels, labels_pred=self.val_cluster_assignments)
        self.log('val/nmi', nmi, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def _enqueue(self, features, labels):
        features = F.normalize(features, dim=1, p=2)
        self.knn_feats.append(features)
        self.knn_labels.append(labels)

    def compute_dino(self, batch):
        """ The input is image only """
        it = self.trainer.global_step
        epoch = self.trainer.current_epoch
        optimizer = self.trainer.optimizers[0]
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]
        self.log(f"train/lr", self.lr_schedule[it])

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
            self.log("dino/last_layer_weight_v_norm", last_layer.weight_v.norm(p=2, dim=1).mean().item())
            self.log("dino/center_mean", self.dino_loss.center.mean().item())
            self.log("dino/center_var", self.dino_loss.center.var().item())
            self.log("dino/teacher_var", teacher_embs.var(dim=0).mean().item())
            self.log("dino/student_var", student_embs.var(dim=0).mean().item())
        return {'loss': loss, 
                'features': teacher_embs[:images[0].size(0), :].detach(),
                'cluster_assignments': teacher_output[:images[0].size(0), :].argmax(-1).detach(),
                'labels': torch.tensor(batch[self.config['dino_label_key']]).long().cuda()}

    def compute_mlm(self, batch):
        text_ids = batch[f"text_ids_mlm"]
        text_labels = batch[f"text_labels_mlm"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        backbone = self.student.backbone
        x = text_embeds
        for blk in backbone.blocks:
            x = blk(x, mask=text_masks)
        x = backbone.norm(x)
        mlm_logits = self.mlm_head(x) / 0.04
        mlm_loss = F.cross_entropy(
                    mlm_logits.view(-1, self.config["vocab_size"]),
                    text_labels.view(-1), ignore_index=-100)
        phase = "train" if self.training else "val"
        self.log(f"mlm/{phase}/loss", mlm_loss.item())
        return {'mlm_loss': mlm_loss}

    def forward(self, batch):
        output = self.compute_dino(batch)
        if self.use_mlm:
            output.update(self.compute_mlm(batch))
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
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
        output = self.forward(batch)
        test_labels = output['labels']
        test_features = output['features']
        test_features = F.normalize(test_features, dim=1, p=2)
        top1, top5 = do_knn_step(self.knn_feats.T, self.knn_labels, test_features, test_labels,
                                           T=self.config['knn_temp'], k=self.config['nb_knn'], num_classes=1000)
        self.log(f'val/knn_top1', top1, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f'val/knn_top5', top5, on_step=False, on_epoch=True, sync_dist=True)

        # Compute NMI
        self.val_labels.append(test_labels)
        self.val_cluster_assignments.append(output['cluster_assignments'])

    def configure_optimizers(self):
        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(self.student)
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        return optimizer
