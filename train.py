import time
import copy
import numpy as np
import math
import matplotlib.pyplot as plt

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

# TODO: track training / test time

# TODO: visualize training and test losses

def plot_train_test_loss(epochs, train_loss, test_loss):
    """Plot the average train and testloss per epoch on a line plot."""
    # Actually starts at 1.
    x = range(epochs)

    # plot lines
    plt.plot(x, train_loss, label = "train loss")
    plt.plot(x, test_loss, label = "test loss")
    plt.legend()
    plt.savefig(f'train_test_loss_{epochs}epochs')
    plt.show()

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

    start_time = time.time()

    train_losses = []
    test_losses = []

    best_loss = 99
    n_epochs = 5 # 100
    for epoch in range(n_epochs):
        begin_epoch = time.time()

        epoch_iter = 0
        train_loss = 0
        for i, data in enumerate(trainloader):
            # print(f"Getting batch number {i}")
            iter_start_time = time.time()
            epoch_iter += opt.batch_size
            batch_amount = 0

            # data contains multiple point clouds!
            # print(f"Data length (==batchsize=4): {len(data)}")

            if opt.dataset=='modelnet' or opt.dataset=='shrec':
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            elif opt.dataset=='shapenet' or opt.dataset=='catenary_arches':
                # print('Going to unpack the data of a single batch')
                input_pc, input_label, input_node, input_node_knn_I = data # pc, label, som_node, som_knn_I
                model.set_input(input_pc, input_label, input_node, input_node_knn_I)

            batch_amount += input_label.size()[0]
            # print('About to optimize the model based on current training batch')
            model.optimize()
            # print('After optimizing the model based on current training batch')


            train_loss += model.loss.detach() * input_label.size()[0]

            if i % 10 == 0:
                # print/plot errors
                # t = (time.time() - iter_start_time) / opt.batch_size

                errors = model.get_current_errors()

                # print(model.test_loss.item())
                # print(errors)
                # print()

                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch, i)

        train_loss /= batch_amount

        train_losses.append(model.loss)

        end_train = time.time()
        print(f"Epoch {epoch} took {end_train-begin_epoch} seconds.")

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
                # TODO: why is loss multiplied here?! To average out loss per data sample?
                model.test_loss += model.loss_chamfer.detach() * input_label.size()[0]

            model.test_loss /= batch_amount

            test_losses.append(model.test_loss)

            if model.test_loss.item() < best_loss:
                best_loss = model.test_loss.item()
            print('Tested network. So far lowest loss: %f' % best_loss)

        end_test = time.time()
        print(f"Testing after epoch {epoch} took {end_test-end_train} seconds.")

        # learning rate decay
        if epoch%20==0 and epoch>0:
            model.update_learning_rate(0.5)

        # save network
        if epoch%1==0 and epoch>0:
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)
            model.save_network(model.decoder, 'decoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)

    print(f"Length of all training losses should be equal to number of epochs (5): {len(train_losses)}")
    print(f"Length of all test losses should be equal to number of epochs (5): {len(test_losses)}")

    plot_train_test_loss(n_epochs, train_losses, test_losses)





