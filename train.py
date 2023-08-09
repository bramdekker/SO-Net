import time
import copy
import os
import numpy as np
import math
import laspy
import argparse
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

def avg(l):
    return sum(l) / len(l)

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

def save_to_las(model, o_file_name, p_file_name):
    """Save the current batch of reconstructions to LAS files otogether with the original point clouds."""
    input_pred_dict = model.get_current_visuals()
    input_pc, predicted_pc = input_pred_dict["input_pc"], input_pred_dict["predicted_pc"]

    for i in range(len(input_pc)):
        # Save original point cloud.
        input_data = input_pc[i]
        # 1. Create a new header
        header = laspy.LasHeader(point_format=6, version="1.4")

        # 2. Create a Las
        las = laspy.LasData(header)

        las.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
        las.y = input_data[1]
        las.z = input_data[2]
        # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

        las.write("%s_%d.las" % (o_file_name, i))

        # Save predicted point cloud.
        predicted_data = predicted_pc[i]
        # 1. Create a new header
        header = laspy.LasHeader(point_format=6, version="1.4")

        # 2. Create a Las
        las2 = laspy.LasData(header)

        las2.x = predicted_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
        las2.y = predicted_data[1]
        las2.z = predicted_data[2]
        # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

        las2.write("%s_%d.las" % (p_file_name, i))

def train_loocv(opt):
    """Train and validate the autoencoder with leave-one-out cross validation method."""
    # ~270 million params (PointNet ~ 4M, MVCNN ~ 60M)
    file_list = [f for f in os.listdir(opt.dataroot) if (f.endswith(".las") or f.endswith(".laz"))]
    print_epochs_time = True

    test_losses = []

    for f in file_list:
        trainset = ArchesLoader(opt.dataroot, 'loocv', opt, f)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

        testset = ArchesLoader(opt.dataroot, 'loocv_test', opt, f)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

        model = Model(opt) #.to(device)?
        before_train_mem = torch.cuda.memory_allocated(opt.device)
        print(f"Amount of GPU memory allocated in MB before training (approx. 15.000 available): {before_train_mem / 1000000}")

        train_losses = []

        for epoch in range(opt.epochs):
            begin_epoch = time.time()

            epoch_iter = 0
            train_loss = 0
            batch_amount = 0
            for i, data in enumerate(trainloader):
                epoch_iter += opt.batch_size
                
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data # pc, label, som_node, som_knn_I
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

                batch_amount += input_label.size()[0]
                model.optimize()

                train_loss += model.loss.cpu().data * input_label.size()[0]

                # Save original and reconstructed point cloud of 1st batch of last epoch to files.
                if epoch == opt.epochs - 1 and i == 0:
                    save_to_las(model, "train_%s_loocv_original_pc" % (f), "train_%s_loocv_predicted_pc" % f)

            train_loss /= batch_amount
            train_losses.append(train_loss)

            end_train = time.time()
            if print_epochs_time:
                print(f"Epoch {epoch} took {end_train-begin_epoch} seconds.")
                print_epochs_time = False

            # learning rate decay
            if epoch % opt.lr_decay_step == 0 and epoch > 0:
                model.update_learning_rate(opt.lr_decay_rate)

            # save network
            if epoch%1==0 and epoch>0:
                print("Saving network...")
                model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)
            #     model.save_network(model.decoder, 'decoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)


        # test network
        if epoch >= 0 and epoch%1==0:
            with torch.no_grad():
                batch_amount = 0
                model.test_loss.data.zero_()
                test_loss = 0
                
                for i, data in enumerate(testloader):
                    input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                    model.test_model()

                    batch_amount += input_label.size()[0]
                    test_loss += model.loss.cpu().data * input_label.size()[0]

                test_loss /= batch_amount

                test_losses.append((f, test_loss))
            
                # Save predictions and originals inputs of the testset.
                if epoch == opt.epochs - 1:
                    save_to_las(model, "test_%s_loocv_original_pc" % f, "test_%s_loocv_predicted_pc" % f)

        end_test = time.time()
        print(f"Testing took {end_test-end_train} seconds.")

        # print(f"Length of all training losses should be equal to number of epochs ({opt.epochs}): {len(train_losses)}")
        # print(f"Length of all test losses should be equal to number of epochs ({opt.epochs}): {len(test_losses)}")

        print("Train losses: ", train_losses)
        # print("Test losses: ", test_losses)

        # plot_train_test_loss(opt.epochs, train_losses, test_losses)
    # test_loss_np = np.asarray(test_losses)
    test_names_tuple, test_loss_tuple = zip(*test_losses)
    for i in range(len(test_names_tuple)):
        print(f"Test file {test_names_tuple[i]} has a loss of {test_loss_tuple[i]}", end=' ')
    test_losses_np = np.asarray(test_loss_tuple)
    print(f"The mean test loss is {np.mean(test_losses_np)} and the standard deviation is {np.std(test_losses_np)}")


def train_model(model, dataset, epoch, opt):
    """Train the model on the given dataset and using parameters defined in the Option object opt."""
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads, drop_last=True)

    begin_epoch = time.time()

    train_loss = 0
    batch_amount = 0
    for i, data in enumerate(trainloader):
        # Extract batch input and store it in the model.
        input_pc, input_sn, input_label, input_node, input_node_knn_I = data # pc, sn, label, som_node, som_knn_I
        model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

        # Print out memory for 1st epoch.
        if epoch == 0:
            after_loading_input_mem = torch.cuda.memory_allocated(opt.device)
            print(f"Amount of GPU memory allocated in MB after loading input: {after_loading_input_mem / 1000000}")

        # Get the number of samples in current batch.
        batch_amount += input_label.size()[0]

        # Train model on current batch.
        model.optimize()

        # Get loss for current batch multiplied by number of samples in batch.
        train_loss += model.loss.cpu().data * input_label.size()[0]

        # Save original and reconstructed point cloud of 1st batch of last epoch to files.
        if opt.save_train_pcs and epoch == opt.epochs - 1 and i == 0:
            save_to_las(model, "original_train", "predicted_train")

    # Get average loss per sample.
    train_loss /= batch_amount

    end_train = time.time()

    # Print approx time for epoch once.
    if epoch == 0:
        epoch_time = end_train - begin_epoch
        print(f"A training epoch takes approx {int(epoch_time)} seconds.")
        print(f"Total taining time will be approx. {int(opt.avg_rounds * opt.epochs * epoch_time)} minutes.")

    return train_loss.item()
    

def test_model(model, dataset, epoch, opt):
    """Test the model on the given dataset and using parameters defined in the Option object opt."""
    testloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

    with torch.no_grad():
        model.test_loss.data.zero_()

        begin_epoch = time.time()

        test_loss = 0
        batch_amount = 0
        for i, data in enumerate(testloader):

            # Extract batch input and store it in the model.
            input_pc, input_sn, input_label, input_node, input_node_knn_I = data # pc, sn, label, som_node, som_knn_I
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

            # # Print out memory for 1st epoch.
            # if epoch == 0:
            #     after_loading_input_mem = torch.cuda.memory_allocated(opt.device)
            #     print(f"Amount of GPU memory allocated in MB after loading input: {after_loading_input_mem / 1000000}")

            # Get the number of samples in current batch.
            batch_amount += input_label.size()[0]

            # Test the model.
            model.test_model()

            # Get loss for current batch multiplied by number of samples in batch.
            test_loss += model.loss.cpu().data * input_label.size()[0]

            # Save original and reconstructed point cloud of 1st batch of last epoch to files.
            if opt.save_test_pcs and epoch == opt.epochs - 1 and i == 0:
                save_to_las(model, "original_test", "predicted_test")

        # Get average loss per sample.
        test_loss /= batch_amount

        end_train = time.time()

        # Print approx time for epoch once.
        if epoch == 0:
            epoch_time = end_train - begin_epoch
            print(f"A test epoch takes approx {int(epoch_time)} seconds.")
            print(f"Total testing time will be approx. {int(opt.avg_rounds * opt.epochs * epoch_time)} minutes.")

        return test_loss.item()


def main():
    test_losses = []
    train_losses = []

    train_frac = opt.train_fraction
    test_frac = round(1 - train_frac, 2)
    
    dataset = ArchesLoader(opt.dataroot, 'all', opt)

    training_size = round(train_frac * len(dataset))
    test_size = len(dataset) - training_size
    
    # For every experiment, record lowest loss, all test losses and train losses. Losses are averages per epoch.
    # Lowest loss shape = (opt.avg_rounds) -> Avg lowest loss
    # Test/train losses shape = (opt.avg_rounds, epochs) -> Avg plots for train and test
    for i in range(opt.avg_rounds):
        # Run experiment with randomly initizalized model, training- and test set.
        ae_model = Model(opt)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [training_size, test_size])

        train_losses_round = []
        test_losses_round = []

        best_test_loss = 0.7

        for i in range(opt.epochs):
            # Train models and record training losses
            train_loss_round = train_model(ae_model, train_dataset, i, opt)
            train_losses_round.append(train_loss_round)

            # Test model and record test losses
            test_loss_round = test_model(ae_model, test_dataset, i, opt)
            test_losses_round.append(test_loss_round)

            if test_loss_round < best_test_loss:
                best_test_loss = test_loss_round
                print("Saving network...")
                ae_model.save_network(ae_model.encoder, 'encoder', 'boxes_5_5_5_labeled_%depochs' % opt.epochs, opt.gpu_id)
                ae_model.save_network(ae_model.decoder, 'decoder', 'boxes_5_5_5_labeled_%depochs' % opt.epochs, opt.gpu_id)



        print(f"train_losses from this round are (number of epochs length): {train_losses_round}")
        print(f"test_losses from this round are (number of epochs length): {test_losses_round}")
        # Add recorded train and test losses to arrays.
        train_losses.append(train_losses_round)
        test_losses.append(test_losses_round)

    
    print(f"train_losses for all rounds are (rounds x number of epochs length): {train_losses}")
    print(f"test_losses for all rounds are (rounds x number of epochs length): {test_losses}")

    avg_lowest_test_loss = avg([min(arr) for arr in test_losses])

    print(f"Average lowest test loss is {avg_lowest_test_loss}")

    # TODO: shapes are not equal!!
    avg_train_losses = np.average(np.array(train_losses), axis=0) # should be 25 x 1
    avg_test_losses = np.average(np.array(test_losses), axis=0)

    plot_train_test_loss(opt.epochs, avg_train_losses, avg_test_losses)


    # """Main function that executes the regular training and testing."""
    # if opt.dataset=='modelnet' or opt.dataset=='shrec':
    #     trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    #     dataset_size = len(trainset)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    #     print('#training point clouds = %d' % len(trainset))

    #     testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    # elif opt.dataset=='shapenet':
    #     trainset = ShapeNetLoader(opt.dataroot, 'train', opt)
    #     dataset_size = len(trainset)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    #     print('#training point clouds = %d' % len(trainset))
    #     testset = ShapeNetLoader(opt.dataroot, 'test', opt)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    # elif opt.dataset=='catenary_arches':
    #     if opt.loocv:
    #         train_loocv(opt)
    #         return

    #     trainset = ArchesLoader(opt.dataroot, 'train', opt)
    #     dataset_size = len(trainset)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    #     print('#training point clouds = %d' % len(trainset))

    #     testset = ArchesLoader(opt.dataroot, 'test', opt)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    #     # print('#test point clouds = %d' % len(testset))
    # else:
    #     raise Exception('Dataset error.')

    # # ~270 million params (PointNet ~ 4M, MVCNN ~ 60M)
    # model = Model(opt) #.to(device)?
    # count_parameters(model.decoder.conv_decoder.deconv1.conv)
    # count_parameters(model.decoder.conv_decoder.deconv1.up_sample)


    # # pytorch_total_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    # # pytorch_total_decoder_params = sum(p.numel() for p in model.decoder.parameters())
    # # print(f"Total number of parameters in encoder ({pytorch_total_encoder_params}) and decoder ({pytorch_total_decoder_params}): {pytorch_total_encoder_params + pytorch_total_decoder_params}")

    # # pytorch_train_encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    # # pytorch_train_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    # # print(f"Total number of trainable parameters in encoder ({pytorch_train_encoder_params}) and decoder ({pytorch_train_decoder_params}): {pytorch_train_encoder_params + pytorch_train_decoder_params}")

    # before_train_mem = torch.cuda.memory_allocated(opt.device)
    # print(f"Amount of GPU memory allocated in MB before training (approx. 15.000 available): {before_train_mem / 1000000}")

    # # visualizer = Visualizer(opt)

    # # print('About to start training loop')

    # start_time = time.time()

    # train_losses = []
    # test_losses = []

    # best_loss = 99
    # for epoch in range(opt.epochs):
    #     begin_epoch = time.time()

    #     epoch_iter = 0
    #     train_loss = 0
    #     batch_amount = 0
    #     for i, data in enumerate(trainloader):
    #         # for j, pc in enumerate(input_pc):
    #         #     data_idx = (i * 4 + j) // 6
    #         #     augment_idx = (i * 4 + j) % 6
    #         #     # print(f"Data idx {data_idx} and augmentation idx {augment_idx} in train.py")

    #         #     header = laspy.LasHeader(point_format=6, version="1.4")
    #         #     #header.offsets = np.min(my_data, axis=0)

    #         #     # 2. Create a Las
    #         #     las = laspy.LasData(header)

    #         #     # print(f"Original pc shape is {pc.shape}")
    #         #     # print(f"Pc first 10 is {pc[0][:10]}")

    #         #     las.x = pc.numpy()[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #         #     las.y = pc.numpy()[1]
    #         #     las.z = pc.numpy()[2]
    #         #     # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #         #     las.write("original_pc_%d_%d.las" % (data_idx, augment_idx))

    #         # continue


    #         # print(f"Getting batch number {i}")
    #         iter_start_time = time.time()
    #         epoch_iter += opt.batch_size

    #         # data contains multiple point clouds!
    #         # print(f"Data length (==batchsize=4): {len(data)}")

    #         if opt.dataset=='modelnet' or opt.dataset=='shrec':
    #             input_pc, input_sn, input_label, input_node, input_node_knn_I = data
    #             model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
    #         elif opt.dataset=='shapenet' or opt.dataset=='catenary_arches':
    #             # print('Going to unpack the data of a single batch')
    #             input_pc, input_sn, input_label, input_node, input_node_knn_I = data # pc, label, som_node, som_knn_I
    #             model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
    #             if epoch == 0:
    #                 after_loading_input_mem = torch.cuda.memory_allocated(opt.device)
    #                 print(f"Amount of GPU memory allocated in MB after loading input: {after_loading_input_mem / 1000000}")

    #         # Check shapes of loaded data.
    #         # print(f"Shape of input_pc is (3x4096) {input_pc.shape}")
    #         # print(f"Shape of input_sn is (3x4096) {input_sn.shape}")
    #         # print(f"Shape of input_node is (3x64) {input_node.shape}")
    #         # print(f"Shape of input_node_knn_I is (64x9) {input_node_knn_I.shape}")

    #         time.sleep(2)


    #         batch_amount += input_label.size()[0]
    #         # print(f"Input label.size()[0] is {input_label.size()[0]} ")
    #         # print('About to optimize the model based on current training batch')
    #         model.optimize()
    #         # print('After optimizing the model based on current training batch')


    #         train_loss += model.loss.cpu().data * input_label.size()[0]

    #         # print("After added training loss")

    #         # if i % 10 == 0:
    #             # print/plot errors
    #             # t = (time.time() - iter_start_time) / opt.batch_size

    #             # errors = model.get_current_errors()

    #             # print(model.test_loss.item())
    #             # print(errors)
    #             # print()

    #             # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    #             # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

    #             # print(model.autoencoder.encoder.feature)
    #             # visuals = model.get_current_visuals()
    #             # visualizer.display_current_results(visuals, epoch, i)

    #         # Save original and reconstructed point cloud of 1st batch of last epoch to files.
    #         if epoch == opt.epochs - 1 and i == 0:
    #             input_pred_dict = model.get_current_visuals()
    #             input_pc, predicted_pc = input_pred_dict["input_pc"], input_pred_dict["predicted_pc"]
    #             # print(f"Length of input entry is {len(input_pc)} (should be {opt.batch_size})")

    #             for i in range(len(input_pc)):
    #                 # Save original point cloud.
    #                 input_data = input_pc[i]
    #                 # 1. Create a new header
    #                 header = laspy.LasHeader(point_format=6, version="1.4")
    #                 #header.offsets = np.min(my_data, axis=0)

    #                 # 2. Create a Las
    #                 las = laspy.LasData(header)

    #                 las.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #                 las.y = input_data[1]
    #                 las.z = input_data[2]
    #                 # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #                 las.write("original_pc_train_%d.las" % i)

    #                 # Save predicted point cloud.
    #                 predicted_data = predicted_pc[i]
    #                 # 1. Create a new header
    #                 header = laspy.LasHeader(point_format=6, version="1.4")
    #                 #header.offsets = np.min(my_data, axis=0)

    #                 # 2. Create a Las
    #                 las2 = laspy.LasData(header)

    #                 las2.x = predicted_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #                 las2.y = predicted_data[1]
    #                 las2.z = predicted_data[2]
    #                 # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #                 las2.write("predicted_pc_train_%d.las" % i)


    #     train_loss /= batch_amount
    #     # print(f"Batch amount is {batch_amount}")

    #     train_losses.append(train_loss)

    #     end_train = time.time()
    #     print(f"Epoch {epoch} took {end_train-begin_epoch} seconds.")

    #     # test network
    #     if epoch >= 0 and epoch%1==0:
    #         with torch.no_grad():
    #             batch_amount = 0
    #             model.test_loss.data.zero_()
    #             test_loss = 0
    #             for i, data in enumerate(testloader):
    #                 if opt.dataset == 'modelnet' or opt.dataset=='shrec':
    #                     input_pc, input_sn, input_label, input_node, input_node_knn_I = data
    #                     model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
    #                 elif opt.dataset == 'shapenet' or 'catenary_arches':
    #                     input_pc, input_sn, input_label, input_node, input_node_knn_I = data
    #                     model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
    #                 model.test_model()

    #                 batch_amount += input_label.size()[0]

    #                 # accumulate loss
    #                 # model.test_loss += model.loss_chamfer.detach() * input_label.size()[0]
    #                 test_loss += model.loss.cpu().data * input_label.size()[0]

    #                 # print(f"TEST: Input_label.size()[0] is {input_label.size()[0]}")

    #             # model.test_loss /= batch_amount
    #             test_loss /= batch_amount
    #             # print(f"TEST: Batch amount is {batch_amount}")

    #             # test_losses.append(model.test_loss.cpu().item())
    #             test_losses.append(test_loss)

    #             if test_loss < best_loss:
    #                 best_loss = test_loss
    #             # if model.test_loss.item() < best_loss:
    #             #     best_loss = model.test_loss.item()
    #             print('Tested network. So far lowest loss: %f' % best_loss)
            
    #             # Save predictions and originals inputs of the testset.
    #             if epoch == opt.epochs - 1:
    #                 input_pred_dict = model.get_current_visuals()
    #                 input_pc, predicted_pc = input_pred_dict["input_pc"], input_pred_dict["predicted_pc"]
    #                 # print(f"Length of input entry is {len(input_pc)} (should be {opt.batch_size})")

    #                 for i in range(len(input_pc)):
    #                     # Save original point cloud.
    #                     input_data = input_pc[i]
    #                     # 1. Create a new header
    #                     header = laspy.LasHeader(point_format=6, version="1.4")
    #                     #header.offsets = np.min(my_data, axis=0)

    #                     # 2. Create a Las
    #                     las = laspy.LasData(header)

    #                     las.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #                     las.y = input_data[1]
    #                     las.z = input_data[2]
    #                     # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #                     las.write("original_pc_%d.las" % i)

    #                     # Save predicted point cloud.
    #                     predicted_data = predicted_pc[i]
    #                     # 1. Create a new header
    #                     header = laspy.LasHeader(point_format=6, version="1.4")
    #                     #header.offsets = np.min(my_data, axis=0)

    #                     # 2. Create a Las
    #                     las2 = laspy.LasData(header)

    #                     las2.x = predicted_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #                     las2.y = predicted_data[1]
    #                     las2.z = predicted_data[2]
    #                     # las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #                     las2.write("predicted_pc_%d.las" % i)

    #     end_test = time.time()
    #     print(f"Testing after epoch {epoch} took {end_test-end_train} seconds.")

    #     # learning rate decay
    #     if epoch%opt.lr_decay_step==0 and epoch>0:
    #         model.update_learning_rate(opt.lr_decay_rate)

    #     # save network
    #     if epoch%1==0 and epoch>0:
    #         print("Saving network...")
    #         model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)
    #         model.save_network(model.decoder, 'decoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)

    # print(f"Length of all training losses should be equal to number of epochs ({opt.epochs}): {len(train_losses)}")
    # print(f"Length of all test losses should be equal to number of epochs ({opt.epochs}): {len(test_losses)}")

    # print("Train losses: ", train_losses)
    # print("Test losses: ", test_losses)

    # plot_train_test_loss(opt.epochs, train_losses, test_losses)


# Get full dataset and then split in train- and testloader.
# Split dataset 70 for autoencoder (11 samples), 15 supervised training segmenter (2 samples), 15 validation segmenter (2 samples).
# Need to make sure that the data samples are non-overlapping, or can autoencoder be trained on all data?
# Normalize data: normalized to be zero-mean to range [-1, 1]
# Use augmentation (rotation, downsampling, translation) to get more data.
# What do I want to test? Train autoencoder on a lot of data, also augmented.
# The generated clusters should be good so that they can be labeled by a user (better to have smaller clusters, then big)
if __name__=='__main__':
    main()
