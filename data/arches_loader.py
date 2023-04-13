import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .augmentation import *


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


# Train/test split is hardcoded in file.
def make_dataset_shapenet_normal(root, mode):
    if mode == 'train':
        f = open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r')
        file_name_list = json.load(f)
        f.close()
    elif mode == 'test':
        f = open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r')
        file_name_list = json.load(f)
        f.close()
    else:
        raise Exception('Mode should be train/test.')

    return file_name_list


# Hardcode the files for ease.
def make_dataset_arches(root, mode):
    if mode == 'train':
        # file_name_list = ["01_01.laz", "01_02.laz", "01_03.laz", "02_01.laz", "02_02.laz", "02_03.laz", "02_04.laz", "03_01.laz", "03_02.laz", "03_03.laz", "03_04.laz"]
        file_name_list = ["01_01_norm.laz", "01_02_norm.laz", "01_03_norm.laz", "02_01_norm.laz", "02_02_norm.laz", "02_03_norm.laz", "02_04_norm.laz", "03_01_norm.laz", "03_02_norm.laz", "03_03_norm.laz", "03_04_norm.laz"]
    elif mode == 'test':
        file_name_list = ["04_01_norm.laz", "04_02_norm.laz", "04_03_norm.laz", "04_04_norm.laz"]

        # file_name_list = ["01_01.laz", "01_02.laz", "01_03.laz", "02_01.laz", "02_02.laz", "02_03.laz", "02_04.laz", "03_01.laz", "03_02.laz", "03_03.laz", "03_04.laz"]
    else:
        raise Exception('Mode should be train/test.')

    return file_name_list


class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class ArchesLoader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(ArchesLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # Parameters for SOM
        self.node_num = opt.node_num
        self.rows = round(math.sqrt(self.node_num))
        self.cols = self.rows

        self.noise_value = 0.000001
        self.rotations = [-30, 0, 30]

        # Add random noise: what value to base it on? Normalized data, biggest max-min difference?

        # Normalize? Need max and min of x, y and z

        # self.dataset = make_dataset_shapenet_normal(self.root, self.mode) # list of file names
        self.dataset = make_dataset_arches(self.root, self.mode) # list of file names
        # ensure there is no batch-1 batch
        # if len(self.dataset) % self.opt.batch_size == 1:
        #     self.dataset.pop()

        # load the folder-category txt
        # self.categories = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop',
        #                    'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
        # self.folders = ['02691156', '02773838', '02954340', '02958343', '03001627', '03261776', '03467517', '03624134',
        #                 '03636649', '03642806', '03790512', '03797390', '03948459', '04099429', '04225987', '04379243']

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    # Add augmentation entries here as well. 
    def __len__(self):
        return 2 * len(self.rotations) * len(self.dataset)

    def __getitem__(self, index):
        # pointnet++ dataset
        # Get SOM file
        dataset_idx = index // 6 # 0-len(self.dataset)
        augmentation_idx = index % 6 # 0-5 


        # if self.mode == "test":
            # print(f"Length of dataset is (24): {len(self.dataset)}")
            # print(f"Dataset index is (0-3): {dataset_idx}")
        file = self.dataset[dataset_idx][0:-4]
        # print(f"Just before np load of npz file {file}")
        data = np.load(os.path.join(self.root, '%dx%d' % (self.rows, self.cols), file + '.npz'))

        # Data is like [[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]], so a Nx3 array
        # pc_np = np.transpose(data['pc'])

        # Data is like [[x1, x2, ..., xn], [y1, y2, ..., yn], [z1, z2, ..., zn]], so a 3xN array
        pc_np = data['pc']
        sn_np = np.zeros_like(pc_np)

        if self.opt.surface_normal:
            sn_np = data['sn']
        # seg_np = data['part_label']
        som_node_np = data['som_node'] # 3 x 64
        # som_node_np = np.transpose(data['som_node']) #3x64
        # print(f"Inital som_node_np shape is {som_node_np.shape}")

        label = 1 # dummy value, always 1
        # label = self.folders.index(file[0:8])
        # assert(label >= 0)

        # print("Just before downsampling")
        # Downsample to the number of input points specified in options. Also sn when not used!
        if self.opt.input_pc_num < pc_np.shape[1]:
            # print(f"pc_np.shape: {pc_np.shape}")
            chosen_idx = np.random.choice(pc_np.shape[1], self.opt.input_pc_num, replace=False)
            # pc_np = pc_np[chosen_idx, :]
            pc_np = pc_np[:, chosen_idx]
            sn_np = sn_np[:, chosen_idx]
            # seg_np = seg_np[chosen_idx]
        else:
            # print(f"In else: pc_np.shape: {pc_np.shape}")
            chosen_idx = np.random.choice(pc_np.shape[1], self.opt.input_pc_num-pc_np.shape[1], replace=True)
            # pc_np_redundent = pc_np[chosen_idx, :]
            pc_np_redundent = pc_np[:, chosen_idx]
            sn_np_redundent = sn_np[: chosen_idx]
            # seg_np_redundent = seg_np[chosen_idx]
            pc_np = np.concatenate((pc_np, pc_np_redundent), axis=0) # Ux3 concat Vx3 -> Nx3
            sn_np = np.concatenate((sn_np, sn_np_redundent), axis=0)
            # seg_np = np.concatenate((seg_np, seg_np_redundent), axis=0)

        # print(f"Shape just before augmentation is {pc_np.shape}")

        # print("Just after downsampling")
        # TODO: check augmentation methods. One of them returns Nx3 data instead of 3xN!
        # augmentation
        if self.mode == 'train':
            # index 1-6: 1-3 without noise, 4-6 with noise
            if augmentation_idx >= 3:
                # print("Adding random noise")
                pc_np = random_noise(pc_np, -self.noise_value, self.noise_value)
                if self.opt.surface_normal:
                    sn_np = random_noise(sn_np, -self.noise_value, self.noise_value)
                som_node_np = random_noise(som_node_np, -self.noise_value, self.noise_value)
                # print("After adding random noise")
                # print(f"pc_np.shape: {pc_np.shape}")
            
            # print(f"Shape after random noise is {som_node_np.shape}")


            # 0, 3 = -45
            # 1, 4 = 0
            # 2, 5 = 45
            rotation_angle = self.rotations[augmentation_idx % len(self.rotations)]
            if rotation_angle != 0:
                # print("Rotating point cloud")
                pc_np = rotate_point_cloud(pc_np, rotation_angle)
                if self.opt.surface_normal:
                    sn_np = rotate_point_cloud(sn_np, rotation_angle)
                som_node_np = rotate_point_cloud(som_node_np, rotation_angle)
                # print("After rotating point cloud")

            # print(f"Shape after rotation is {som_node_np.shape}")

            # rotate by random degree over model z (point coordinate y) axis
            # pc_np = rotate_point_cloud(pc_np)
            # som_node_np = rotate_point_cloud(som_node_np)

            # rotate by 0/90/180/270 degree over model z (point coordinate y) axis
            # pc_np = rotate_point_cloud_90(pc_np)
            # som_node_np = rotate_point_cloud_90(som_node_np)

            # random jittering
            # pc_np = jitter_point_cloud(pc_np)
            # sn_np = jitter_point_cloud(sn_np)
            # som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)

            # print("Scaling point cloud")
            # random scale
            scale = np.random.uniform(low=0.9, high=1.1)
            pc_np = pc_np * scale
            if self.opt.surface_normal:
                sn_np = sn_np * scale
            som_node_np = som_node_np * scale
            # print("After scaling point cloud")

            # random shift
            # shift = np.random.uniform(-0.1, 0.1, (1,3))
            # pc_np += shift
            # som_node_np += shift

        # convert to tensor
        pc = torch.from_numpy(pc_np.astype(np.float32))  # 3xN
        sn = torch.from_numpy(sn_np.astype(np.float32))  # 3xN
        # seg = torch.from_numpy(seg_np.astype(np.int64))  # N

        # som
        som_node = torch.from_numpy(som_node_np.astype(np.float32))  # 3xnode_num
        # print("Tranferred tensors to gpu")

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            # print(f"Shape of some_node_np is {som_node_np.shape}")
            D, I = self.knn_builder.self_build_search(som_node_np.transpose()) # Input is Nx3
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # node_num x som_k
            # print(f"som_knn_I shape (64 x 9): {som_knn_I.shape}")
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape(
                (self.opt.node_num, 1)))  # node_num x 1
        # print("Just before returning")

        # print(f"Final shape of pc is {pc.shape}. Index is {augmentation_idx}")

        return pc, sn, label, som_node, som_knn_I



if __name__=="__main__":
    # dataset = make_dataset_modelnet40('/ssd/dataset/modelnet40_ply_hdf5_2048/', True)
    # print(len(dataset))
    # print(dataset[0])


    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.input_pc_num = 8000
            self.batch_size = 20
            self.node_num = 49
    opt = VirtualOpt()
    trainset = ShapeNetLoader('/ssd/dataset/shapenet_part_seg_hdf5_data/', 'train', opt)
    print(len(trainset))
    pc, seg, label, som_node = trainset[10]

    # print(label)
    print(seg)

    x_np = pc.numpy().transpose()
    node_np = som_node.numpy().transpose()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_np[:, 0].tolist(), x_np[:, 1].tolist(), x_np[:, 2].tolist(), s=1)
    ax.scatter(node_np[:, 0].tolist(), node_np[:, 1].tolist(), node_np[:, 2].tolist(), s=6, c='r')
    plt.show()

