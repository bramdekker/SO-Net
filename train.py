import time
import copy
import numpy as np
import math
import laspy
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
from prettytable import PrettyTable

from models.autoencoder import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from data.arches_loader import ArchesLoader
from data.shapenet_loader import ShapeNetLoader
# from util.visualizer import Visualizer

# TODO: track training / test time
# batch size = 8 / input_pc_num = 16384
# TODO: visualize training and test losses
# Number of parameters in decoder SO-Net (in paper):
# - Conv part: 
#       - 
# - FC part: 27.273.728 params
#       - 1st: 1024 * 2048 (weights) + 2048 (biases) = 2.099.200
#       - 2nd: 2048 * 3072 (weights) + 3072 (biases) = 6.294.528
#       - 3rd: 3072 * 4096 + 4096 = 12.587.008
#       - 4th: 4096 * 512 * 3 + 512 * 3 = 6.292.992

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

def non_shared_parameters(model):
    return sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        testset = ArchesLoader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
        # print('#test point clouds = %d' % len(testset))
    else:
        raise Exception('Dataset error.')

    # ~270 million params (PointNet ~ 4M, MVCNN ~ 60M)
    model = Model(opt)
    count_parameters(model.decoder.conv_decoder.deconv1.conv)
    count_parameters(model.decoder.conv_decoder.deconv1.up_sample)


    pytorch_total_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    pytorch_total_decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total number of parameters in encoder ({pytorch_total_encoder_params}) and decoder ({pytorch_total_decoder_params}): {pytorch_total_encoder_params + pytorch_total_decoder_params}")

    pytorch_train_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    pytorch_train_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in encoder ({pytorch_train_encoder_params}) and decoder ({pytorch_train_decoder_params}): {pytorch_train_encoder_params + pytorch_train_decoder_params}")

    before_train_mem = torch.cuda.memory_allocated(opt.device)
    print(f"Amount of GPU memory allocated in MB before training (approx. 15.000 available): {before_train_mem / 1000000}")

    # visualizer = Visualizer(opt)

    # print('About to start training loop')

    start_time = time.time()

    train_losses = []
    test_losses = []

    best_loss = 99
    for epoch in range(opt.epochs):
        begin_epoch = time.time()

        epoch_iter = 0
        train_loss = 0
        batch_amount = 0
        for i, data in enumerate(trainloader):
            if i >= 3:
                break

            # Go over first 12 ~ 3 batches and just save original (downsampled) pc to check if correct!
            # Only 1_1 is correct, all others is just random noise in unit cube!!! Probably rotation + noise is deforming them. 
            input_pc, input_label, input_node, input_node_knn_I = data # pc, label, som_node, som_knn_I

            for j, pc in enumerate(input_pc):
                data_idx = (i * 4 + j) // 6
                augment_idx = (i * 4 + j) % 6
                # print(f"Data idx {data_idx} and augmentation idx {augment_idx} in train.py")

                header = laspy.LasHeader(point_format=6, version="1.4")
                #header.offsets = np.min(my_data, axis=0)

                # 2. Create a Las
                las = laspy.LasData(header)

                # print(f"Original pc shape is {pc.shape}")
                # print(f"Pc first 10 is {pc[0][:10]}")

                las.x = pc.numpy()[0] # Array with all x coefficients. [x1, x2, ..., xn]
                las.y = pc.numpy()[1]
                las.z = pc.numpy()[2]
                # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

                las.write("original_pc_%d_%d.las" % (data_idx, augment_idx))

            continue


            # print(f"Getting batch number {i}")
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
                after_loading_input_mem = torch.cuda.memory_allocated(opt.device)
                print(f"Amount of GPU memory allocated in MB after loading input: {after_loading_input_mem / 1000000}")

            batch_amount += input_label.size()[0]
            # print(f"Input label.size()[0] is {input_label.size()[0]} ")
            # print('About to optimize the model based on current training batch')
            model.optimize()
            # print('After optimizing the model based on current training batch')


            train_loss += model.loss.cpu().data * input_label.size()[0]

            # if i % 10 == 0:
                # print/plot errors
                # t = (time.time() - iter_start_time) / opt.batch_size

                # errors = model.get_current_errors()

                # print(model.test_loss.item())
                # print(errors)
                # print()

                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch, i)

            after_batch_mem = torch.cuda.memory_allocated(opt.device)
            print(f"Amount of GPU memory allocated in MB after batch training: {after_batch_mem / 1000000}")

        if epoch == 9: # n_epochs - 1
            input_pred_dict = model.get_current_visuals()
            input_pc, predicted_pc = input_pred_dict["input_pc"], input_pred_dict["predicted_pc"]
            print(f"Length of input entry is {len(input_pc)} (should be {opt.batch_size})")

            for i in range(len(input_pc)):
                # Save original point cloud.
                input_data = input_pc[i]
                # 1. Create a new header
                header = laspy.LasHeader(point_format=6, version="1.4")
                #header.offsets = np.min(my_data, axis=0)

                # 2. Create a Las
                las = laspy.LasData(header)

                las.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
                las.y = input_data[1]
                las.z = input_data[2]
                # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

                las.write("original_pc_%d.las" % i)

                # Save predicted point cloud.
                predicted_data = predicted_pc[i]
                # 1. Create a new header
                header = laspy.LasHeader(point_format=6, version="1.4")
                #header.offsets = np.min(my_data, axis=0)

                # 2. Create a Las
                las2 = laspy.LasData(header)

                las2.x = predicted_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
                las2.y = predicted_data[1]
                las2.z = predicted_data[2]
                # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

                las2.write("predicted_pc_%d.las" % i)


        train_loss /= batch_amount
        # print(f"Batch amount is {batch_amount}")

        train_losses.append(train_loss)

        end_train = time.time()
        print(f"Epoch {epoch} took {end_train-begin_epoch} seconds.")

        # test network
        if epoch >= 0 and epoch%1==0:
            with torch.no_grad():
                batch_amount = 0
                model.test_loss.data.zero_()
                test_loss = 0
                for i, data in enumerate(testloader):
                    if opt.dataset == 'modelnet' or opt.dataset=='shrec':
                        input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                        model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                    elif opt.dataset == 'shapenet' or 'catenary_arches':
                        input_pc, input_label, input_node, input_node_knn_I = data
                        model.set_input(input_pc, input_label, input_node, input_node_knn_I)
                    model.test_model()

                    batch_amount += input_label.size()[0]

                    # accumulate loss
                    # model.test_loss += model.loss_chamfer.detach() * input_label.size()[0]
                    test_loss += model.loss.cpu().data * input_label.size()[0]

                    # print(f"TEST: Input_label.size()[0] is {input_label.size()[0]}")

                # model.test_loss /= batch_amount
                test_loss /= batch_amount
                # print(f"TEST: Batch amount is {batch_amount}")

                # test_losses.append(model.test_loss.cpu().item())
                test_losses.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                # if model.test_loss.item() < best_loss:
                #     best_loss = model.test_loss.item()
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

    print(f"Length of all training losses should be equal to number of epochs ({opt.epochs}): {len(train_losses)}")
    print(f"Length of all test losses should be equal to number of epochs ({opt.epochs}): {len(test_losses)}")

    print("Train losses: ", train_losses)
    print("Test losses: ", test_losses)

    plot_train_test_loss(opt.epochs, train_losses, test_losses)





