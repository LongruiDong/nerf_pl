"""
仿照phototourism写replica的接口 注意使用gt pose 和depth对应的点云
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
import open3d as o3d


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.
    copy from nice-slam
    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion() # w x y z https://docs.blender.org/api/current/mathutils.html#mathutils.Quaternion
    if Tquad:
        tensor = np.concatenate([T, quad], 0) # tx y z qw qx qy qz
    else:
        tensor = np.concatenate([quad, T], 0) # QW, QX, QY, QZ, TX, TY, TZ
    # tensor = torch.from_numpy(tensor).float()
    # if gpu_id != -1:
    #     tensor = tensor.to(gpu_id)
    return tensor

class ReplicaGTDataset(Dataset): # 继承了pytorch torch.utils.data.Dataset 的抽象类
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

    def createpcd(self, w2c_mats, skip = 30):
        # 从depth png 得到世界系下 稀疏化后 的点云
        inter = o3d.camera.PinholeCameraIntrinsic()
        inter.set_intrinsics(1200, 680, 600.0, 600.0, 599.5, 339.5)
        pcd_combined = o3d.geometry.PointCloud() # 用于储存多帧合并后的点云 世界系
        for v in range(self.numimg):
            if v%skip != 0 :
                continue
            depthfile = self.depthfiles[v]
            depth_raw = o3d.io.read_image(depthfile) # np.asarray(depth_raw) uint16 680*1200=816000
            pcd_idx = o3d.geometry.PointCloud.create_from_depth_image(
                            depth_raw, inter,
                            extrinsic=w2c_mats[v], #测试这里就用外参 哦好像是得用 w2c!!
                            depth_scale=self.png_depth_scale,
                            stride=50) # 5-(2185582, 3) 10-(546381,3) 20-(136598,3) 50-(22494,3)
            # http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.create_from_depth_image
            pcd_combined += pcd_idx
        
        abox = pcd_combined.get_axis_aligned_bounding_box() # 这个就是原坐标系下bound 和我之前nice-slam同样方法office0.yaml注释一样！
        print('axis_aligned_bounding_box: \n', abox) # AxisAlignedBoundingBox: min: (-2.00841, -3.15475, -1.15338), max: (2.39718, 1.81201, 1.78262)
        pcdarray = np.asarray(pcd_combined.points) # 转为数组
        print('pcd shape: \n', pcdarray.shape) #(n,3) (54639637, 3) colmap: ~20787
        
        return pcdarray # M,3
        
    
    
    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        # tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(self.root_dir[:-1]) # office0/ 
        # 都作为训练 这个列表是colmap auto reconstruction gui 的输出
        # self.files = np.loadtxt(os.path.join(self.root_dir, 'view_imgs.txt'), dtype=str)
        self.files = sorted(
            glob.glob(f'{self.root_dir}/results/frame*.jpg'))
        self.numimg = len(self.files) # 总图像数
        # depth文件名
        self.depthfiles = sorted(glob.glob(f'{self.root_dir}/results/depth*.png'))
        # gt pose 文件
        self.gtposefile = os.path.join(self.root_dir, 'traj.txt')
        self.png_depth_scale = 6553.5 #for depth image in png format uint16 10m内

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cachegt/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cachegt/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            # imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin')) # 其实是colmap的输出
            img_path_to_id = {}
            for v in range(self.numimg):
                img_path_to_id[os.path.basename(self.files[v])] = v # 图像文件名 basename: 其Id  这里从0开始
            self.img_ids = []
            self.image_paths = {} # {id: filename} 
            self.depth_paths = {}
            for filename in list(self.files): # 被重建的图片的id pose 名字 相机模型id  对应kpt在图像的位置 对应3d点id
                id_ = img_path_to_id[os.path.basename(filename)]
                self.image_paths[id_] = os.path.basename(filename)
                self.depth_paths[id_] = os.path.basename(self.depthfiles[id_])
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cachegt/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            # camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32) #都是PINHOLE 改为固定的参数
                # cam = camdata[id_] # 这里的参数 相机内参格式 https://colmap.github.io/format.html#cameras-txt
                img_w, img_h = 1200, 680
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale # Camera(id=1092, model='PINHOLE', width=1013, height=673, params=array([2166.18383789, 2166.18383789,  506.5       ,  336.5       ]))
                K[0, 0] = 600.0*img_w_/img_w # fx # 按照比例 得到新的内参
                K[1, 1] = 600.0*img_h_/img_h # fy
                K[0, 2] = 599.5*img_w_/img_w # cx
                K[1, 2] = 339.5*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cachegt/poses.npy'))
        else:
            c2w_mats = [] # gt pose 已经是 cam->world ！
            w2c_mats = []
            # bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            with open(self.gtposefile, "r") as f:
                lines = f.readlines()
            # fqt = open(os.path.join(self.root_dir,'created/sparse/images.txt'),'a')
            for id_ in self.img_ids: # 前提是id_是从0开始的
                line = lines[id_]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                w2c = np.linalg.inv(c2w)
                # 把w2c 转为 QW, QX, QY, QZ, TX, TY, TZ 存入txt 用来之后colmap fix extrinsic
                quatran = get_tensor_from_camera(w2c)
                # fqt.write("{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} 1 {}\n".format(int(id_+1),  # database 以1开始
                #                     quatran[0],quatran[1],quatran[2],quatran[3],quatran[4],quatran[5],quatran[6],
                #                     str(self.image_paths[id_])))
                # fqt.write("\n") # 空行！
                c2w_mats += [c2w]
                w2c_mats += [w2c]
            # fqt.close()
            c2w_mats = np.stack(c2w_mats, 0) # (N_images, 4, 4)
            w2c_mats = np.stack(w2c_mats, 0)
            self.poses = c2w_mats[:, :3] # (N_images, 3, 4) cam->world
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1 # nice-slam的变换pose也来自于此

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cachegt/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cachegt/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cachegt/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            # pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin')) # https://colmap.github.io/format.html#points3d-txt
            # self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d]) # 所有稀疏3d点 M,3
            # 需要得到gt depth pose 下的点云 所有稀疏3d点 M,3 (54639637, 3) 
            self.xyz_world = self.createpcd(w2c_mats, skip=30)
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1) # (M,4) 齐次表示
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids): #3d点数量相比sparse colmap 太多(~1000x) 导致下面bound计算太慢 
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate 批量转换 (M,4)
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam ,this id 留下的点不一定都能成像吧 更准确的是按照2d-3d对应关系得到此相机的点
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1) # 深度的范围
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max() # 所有深度图的最大值
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor # 把相机在world的平移 按因子缩小
            for k in self.nears: # 深度范围scale 最大不超5m
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor # 3D点同样因子scale
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)} # 把(N,3,4)的pose转为字典
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper) 按照tsv分开训练测试集
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                              if i%50==0 or i%5==0] # 对等于imap kf 才参与优化 这里可以同样实施 to do %5 %50 但两者不能重复
        # 于是上面就有400帧 缩小5倍
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if i%100 == 0]
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test) # 10

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cachegt/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cachegt/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_rgbs = []
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

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8) 即 rayo&d, depth bound(near far), image id  不该是9列？
                                    
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 9) image size 不一定相同 全按行拼起来
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
            return self.N_images_train # 
        if self.split == 'val': # 测试的图片数量
            return self.val_num
        return len(self.poses_test) # 目前没用上

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8], # 数据单元  但每个samples 注意这里的idx是所有ray里的id (8) 
                      'ts': self.all_rays[idx, 8].long(), # 这个才是idx 的ray对应的所属图像id！
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
