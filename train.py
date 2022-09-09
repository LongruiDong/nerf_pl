import os
from opt import get_opts
import torch
from collections import defaultdict
# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse', # coarse 是不用 a t 的！
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min)
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back) # 初始train step 开始别报错

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage): # fit() 先来setup 从trainer.py ln 493 run到此
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if (self.hparams.dataset_name == 'phototourism' or self.hparams.dataset_name == 'replica' 
        or self.hparams.dataset_name == 'replicagt'):
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train) # adam()
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self): # trainer.py l#544 在validation check之后 读入训练数据
        return DataLoader(self.train_dataset,
                          shuffle=False, # 测试 不shuffle  True
                          num_workers=4,
                          batch_size=self.hparams.batch_size, # 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    # 此函数里能看到每一步batch数据的信息
    def training_step(self, batch, batch_nb): # 正式开始 训练1batch from train_loop.py --> #560 run train epoch trainer.py
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts'] # (1024,8) (1024,3) (1024)
        results = self(rays, ts) # 这一步出错！
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb): # 刚开始训练前会来这里 Validation sanity check 验证健全性检查
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts'] # (1,12750,8) (,,3) (1,12750) 12750=rawsize/scale^2
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze() # (H*W) 上面这里是val 故是一张图里的ray等数据
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0: # 就是0 但没找到哪里赋的值
            if (self.hparams.dataset_name == 'phototourism' or self.hparams.dataset_name == 'replica'
                or self.hparams.dataset_name == 'replicagt'):
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W) render的图
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step) # tensorboard能可视化  http://localhost:6006/ 

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log # 目前每1epoch完就会 val一次

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                               '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    print(hparams)
    main(hparams)