"""
仿照phototourism写replica的接口 同样是跟colmap交互
"""
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms as T
# -*- coding:utf-8 -*-
from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

class ReplicaDataset(Dataset): # 继承了pytorch torch.utils.data.Dataset 的抽象类
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        # tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(self.root_dir[:-1]) # office0/ 
        # 都作为训练 这个列表是colmap gui auto recons 的输出
        self.files = np.loadtxt(os.path.join(self.root_dir, 'view_imgs.txt'), dtype=str) # 因为下面有根据colmap来判断 这里就换做全部文件ok
        # self.files = self.files[~self.files['id'].isnull()] # remove data without id
        # self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/1/images.bin')) # 其实是colmap的输出
            img_path_to_id = {} # https://colmap.github.io/format.html#images-txt
            for v in imdata.values():
                img_path_to_id[v.name] = v.id # 图像文件名 basename: 其Id  是从1开始！ 确实colmap 不论auto 还是手动重建 都出现frame001728.jpg没有被重建 images.bin 只有1999帧
            self.img_ids = []
            self.image_paths = {} # {id: filename} 
            for filename in list(self.files): # 被重建的图片的id pose 名字 相机模型id  对应kpt在图像的位置 对应3d点id
                if not filename in img_path_to_id.keys(): # 该帧不在comlmap的输出！ 跳过
                    continue
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/1/cameras.bin')) #实际就1个内参
            for id_ in self.img_ids: # 从1开始的！
                K = np.zeros((3, 3), dtype=np.float32) #都是PINHOLE 要限制我运行colmap时的设置  
                cam = camdata[1] # 这里的参数 相机内参格式 https://colmap.github.io/format.html#cameras-txt
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2) # cx cy为啥先x2  不过x2后 正好是cam参数里的size
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale # Camera(id=1092, model='PINHOLE', width=1013, height=673, params=array([2166.18383789, 2166.18383789,  506.5       ,  336.5       ]))
                K[0, 0] = cam.params[0]*img_w_/img_w # fx # 按照比例 得到新的内参
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_] # QW, QX, QY, QZ, TX, TY, TZ 当使用sparse/1的结果时 看过pose是和created/sparse/images.txt里的gt qt一样
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) 取逆得到cam->world
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1 # nice-slam的变换pose也来自于此

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
            bounds = []
            for i, id_ in enumerate(self.img_ids):
                near_i = self.nears[id_]
                far_i = self.fars[id_]
                bounds += [np.array([near_i, far_i])] 
            self.bounds = np.stack(bounds, 0) # N,2 
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/1/points3D.bin')) # https://colmap.github.io/format.html#points3d-txt
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d]) # 所有稀疏3d点 M,3
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1) # (M,4) 齐次表示
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate 批量转换 (M,4)
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam ,this id 留下的点不一定都能成像吧 更准确的是按照2d-3d对应关系得到此相机的点
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1) # 深度的范围
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max() # 所有深度图的最大值 colmap估计的深度偏大14m sparse/1 最大4.6正常
            scale_factor = max_far/5 # so that the max far is scaled to 5 我觉得这里对于indoor不必要
            scale_factor = 1.
            print('scale-factor: {}'.format(scale_factor)) # 0.9205760955810547
            self.poses[..., 3] /= scale_factor # 把相机在world的平移 按因子缩小
            for k in self.nears: # 深度范围scale 最大不超5m
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor # 3D点同样scale
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)} # 把(N,3,4)的pose转为字典
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper) 按照tsv分开训练测试集
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                              if i%50==0 or i%5==0] # 对等于imap kf 才参与优化 这里可以同样实施 to do %5 %50 但两者不能重复
        # 于是上面就有最多400帧 缩小5倍
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if i%100 == 0]
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test) # 10

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
                count = 0
                for id_ in self.img_ids_train:
                    c2w = torch.FloatTensor(self.poses_dict[id_]) # (3,4)

                    img = Image.open(os.path.join(self.root_dir, 'images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = self.transform(img) # (3, h, w) float32
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img]
                    
                    directions = get_ray_directions(img_h, img_w, self.Ks[id_]) # (h,w,3)
                    rays_o, rays_d = get_rays(directions, c2w) # (h*w,3)
                    rays_t = id_ * torch.ones(len(rays_o), 1) # 记录ray的 image id (h*w,1)
                    rays_pixid =  ( torch.arange(0,int(img_h*img_w),1) + int(count*img_h*img_w) ).unsqueeze(-1) # 记录每个ray global id 用于验证batch数据构成
                    count = count + 1
                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                # rays_pixid],
                                                1)] # (h*w, 8) 即 rayo&d, depth bound(near far), image id  不该是9列？
                                    
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8) image size 不一定相同 全按行拼起来
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays) # 训练时是以ray为样本单位
        if self.split == 'test_train': # 以图片为单位 因为要render 若干图
            return self.N_images_train # 773-10
        if self.split == 'val': # 测试的图片数量
            return self.val_num
        return len(self.poses_test) # 目前没用上

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8], # 数据单元  但每个samples 注意这里的idx是所有ray里的id (8) 
                      'ts': self.all_rays[idx, 8].long(), # 这个才是idx 的ray对应的所属图像id！
                    #   'globalid': self.all_rays[idx, 9].long(), # 看过实际batch里的顺序 的确被shuffle控制
                      'rgbs': self.all_rgbs[idx]} # (3)

        elif self.split in ['val', 'test_train']: # test_train是测试的 那val ? 另一种测试方式？
            sample = {}
            if self.split == 'val':
                id_ = self.val_id # 默认的val img 还是首个训练img
            else: # 'test_train'  就还指定ind拿出 还是来自于训练集 此时idx才是对应于img id
                id_ = self.img_ids_train[idx]
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        else: # 这里目前还没用 因为原论文 是在组半边训练 右半边测试 但此实现还是在训练图上测试的
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
