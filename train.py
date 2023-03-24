import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.autoencoder import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from data.arches_loader import ArchesLoader
from data.shapenet_loader import ShapeNetLoader
# from util.visualizer import Visualizer

# Get full dataset and then split in train- and testloader.
# Split dataset 70 for autoencoder (11 samples), 15 supervised training segmenter (2 samples), 15 validation segmenter (2 samples).
# Need to make sure that the data samples are non-overlapping, or can autoencoder be trained on all data?
# Normalize data: normalized to be zero-mean to range [-1, 1]
# Use augmentation (rotation, downsampling, translation) to get more data.
# What do I want to test? Train autoencoder on a lot of data, also augmented.
# The generated clusters should be good so that they can be labeled by a user (better to have smaller clusters, then big)
if __name__=='__main__':
    if opt.dataset=='modelnet' or opt.dataset=='shrec':
        trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    elif opt.dataset=='shapenet':
        trainset = ShapeNetLoader(opt.dataroot, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        # tesetset = ShapeNetLoader(opt.dataroot, 'test', opt)
        # testloader = torch.utils.data.DataLoader(tesetset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
        testset = ShapeNetLoader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    elif opt.dataset=='catenary_arches':
        trainset = ArchesLoader(opt.dataroot, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        testset = ArchesLoader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
        # print('#test point clouds = %d' % len(testset))
    else:
        raise Exception('Dataset error.')

    model = Model(opt)

    # visualizer = Visualizer(opt)

    # print('About to start training loop')

    best_loss = 99
    for epoch in range(50):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            # data contains multiple point clouds!
            # print(f"Data length (==batchsize=4): {len(data)}")

            if opt.dataset=='modelnet' or opt.dataset=='shrec':
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            elif opt.dataset=='shapenet' or opt.dataset=='catenary_arches':
                # print('Going to unpack the data of a single batch')
                input_pc, input_label, input_node, input_node_knn_I = data # pc, label, som_node, som_knn_I
                model.set_input(input_pc, input_label, input_node, input_node_knn_I)

            # print('About to optimize the model based on current training batch')
            model.optimize()
            # print('After optimizing the model based on current training batch')

            if i % 10 == 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size

                errors = model.get_current_errors()
                
                print(model.test_loss.item())
                print(errors)
                print()

                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss.data.zero_()
            for i, data in enumerate(testloader):
                if opt.dataset == 'modelnet' or opt.dataset=='shrec':
                    input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                elif opt.dataset == 'shapenet' or 'catenary_arches':
                    input_pc, input_label, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_label, input_node, input_node_knn_I)
                model.test_model()

                batch_amount += input_label.size()[0]

                # # accumulate loss
                model.test_loss += model.loss_chamfer.detach() * input_label.size()[0]

            model.test_loss /= batch_amount
            if model.test_loss.item() < best_loss:
                best_loss = model.test_loss.item()
            print('Tested network. So far lowest loss: %f' % best_loss )

        # learning rate decay
        if epoch%20==0 and epoch>0:
            model.update_learning_rate(0.5)

        # save network
        if epoch%1==0 and epoch>0:
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)
            model.save_network(model.decoder, 'decoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)





