# -*- coding:utf-8 -*-
"""
仿照prepare_phototourism.py 来生成replicagt训练数据
"""
import argparse
from datasets import ReplicaGTDataset
import numpy as np
import os
import pickle

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1, # 1200 600
                        help='how much to downscale the images for Replica dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()
    os.makedirs(os.path.join(args.root_dir, 'cachegt'), exist_ok=True)
    print(f'Preparing cachegt for scale {args.img_downscale}...')
    dataset = ReplicaGTDataset(args.root_dir, 'train', args.img_downscale)
    # save img ids
    with open(os.path.join(args.root_dir, f'cachegt/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(args.root_dir, f'cachegt/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.root_dir, f'cachegt/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)
    # save scene points 是scale过的
    np.save(os.path.join(args.root_dir, 'cachegt/xyz_world.npy'),
            dataset.xyz_world)
    # save poses 是scale过的
    np.save(os.path.join(args.root_dir, 'cachegt/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.root_dir, f'cachegt/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_dir, f'cachegt/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs 这里的才是 训练集的图像 不是2000 而是像nie-slam那样参与优化的kf
    np.save(os.path.join(args.root_dir, f'cachegt/rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cachegt/rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())
    print(f"Data cachegt saved to {os.path.join(args.root_dir, 'cachegt')} !")