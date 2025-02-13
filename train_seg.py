import time
import copy
import numpy as np
import math
import laspy

from options_seg import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

from torcheval.metrics import MulticlassConfusionMatrix
from models import losses
from models.segmenter import Model
from data.arches_loader import ArchesLoader


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
    plt.savefig(f'so_baseline_{epochs}epochs')
    plt.show()

def get_overall_acc(conf_matrix):
    total = 0
    truth = 0

    dim = conf_matrix.shape[0]

    for i in range(dim):
        for j in range(dim):
            if i == j:
                truth += conf_matrix.data[i][j]
            
            total += conf_matrix.data[i][j]

    return truth / total

def get_iou(conf_matrix):
    dim = conf_matrix.shape[0]

    ious = []

    for cl in range(dim):
        tp = conf_matrix.data[cl][cl]

        falses = 0
        for j in range(dim):
            if j != cl:
                falses += conf_matrix.data[cl][j]
                falses += conf_matrix.data[j][cl]

        if tp + falses == 0:
            ious.append(0)
        else:
            ious.append(tp / (tp + falses))

    print(ious)

    return ious

def save_to_las(input_pc, pred_labels, orig_labels, save_dir, index):
    """Save the current batch of reconstructions to LAS files otogether with the original point clouds."""
    # ([('pc_colored_predicted', [input_pc_np, pc_color_np]),
    #   ('pc_colored_gt',        [input_pc_np, gt_pc_color_np])])

    # print(f"Shape of the predicted labels is {pred_labels.shape} (should be N).")
    assert(pred_labels.shape == orig_labels.shape)
    
    # This will transform the labels to rgb colors.
    # input_pred_dict = model.get_current_visuals()
    # input_pc, predicted_pc = input_pred_dict["pc_colored_predicted"], input_pred_dict["pc_colored_gt"]

    # Save coordinates + predicted labels to las.

    # Save point cloud with predicted labels.
    input_data = input_pc
    # 1. Create a new header
    header = laspy.LasHeader(point_format=6, version="1.4")

    # 2. Create a Las
    las = laspy.LasData(header)

    # print(f"Shape of input_data is {input_data.shape}")
    # print(f"First two elements of input_data are {input_data[:2]}.")
    las.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    las.y = input_data[1]
    las.z = input_data[2]
    las.classification = pred_labels.squeeze() # Set labels of every point.

    las.write("%s/%s_%d.las" % (save_dir, "predicted", index))

    # 2. Create a Las
    las2 = laspy.LasData(header)

    # print(f"Shape of input_data is {input_data.shape}")
    # print(f"First two elements of input_data are {input_data[:2]}.")
    las2.x = input_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    las2.y = input_data[1]
    las2.z = input_data[2]
    las2.classification = orig_labels.squeeze() # Set labels of every point.

    las2.write("%s/%s_%d.las" % (save_dir, "original", index))



def test_model(model, validationset, save_dir, opt):
    # dataset = ArchesLoader(opt.dataroot, 'all', opt)
    dataloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=opt.nThreads)

    metric = MulticlassConfusionMatrix(opt.classes)

    for i, data in enumerate(dataloader):
        # Get prediction for this batch
        input_pc, input_sn, input_seg, input_node, input_node_knn_I = data #input_label, 
        model.set_input(input_pc, input_sn, input_seg, input_node, input_node_knn_I) #input_label, 

        # Save all prediction per sample to separate file.
        model.test_model()

        if i == 0:
            print(f"Shape of model.score_segmenter.data is {model.score_segmenter.data.shape}")

        for j in range(len(input_pc)):
        # print(f"The shape of the data in model.score_segmenter is {model.score_segmenter.data.shape}") # BxCxN
        # print(f"The first batch entry the first index has length {model.score_segmenter.data[0][0].size()} and looks like {model.score_segmenter.data[0][0]}") 
            _, predicted_seg = torch.max(model.score_segmenter.data[j], dim=0, keepdim=False)
            # print(f"predicted seg shape is {predicted_seg.shape} should be BxNx1 or Bx1xN") # for every point a class
            # print(f"The first two predictions are {predicted_seg[:2]}")

            # if i == 0:
                # print(f"Shape of predicted_seg is {predicted_seg.shape} and shape of input_seg is {input_seg.shape}")
                # print(f"Predicted seg device is {predicted_seg.get_device()}, input_seg device is {input_seg.get_device()}")

            metric.update(predicted_seg.cpu(), input_seg[j])

            # Save 100 samples.
            if i < 100:
                save_to_las(input_pc[j].numpy(), predicted_seg.cpu().numpy(), input_seg[j].numpy(), save_dir, i)

    # Get accuracy and mean IoU for all data.    
    conf_matrix = metric.compute()
    ious = get_iou(conf_matrix)
    print(f"Overall accuracy is {get_overall_acc(conf_matrix * 100)} and the mean IoU is {avg(ious) * 100}")
    
    for class_num, mIoU in enumerate(ious):
        print(f"Class {class_num} had a mIoU of {mIoU * 100}")

# TODO: set labels 11 and 13 to 0 and then use only labels 0 to 13 (e.g. translate labels 14, 15, 16 to 11, 12 and 13 resp.)
# This will increase performance of model quite a bit probably.
def train_model(model, trainset, validationset, opt):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    testloader = torch.utils.data.DataLoader(validationset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

    train_losses_round = [] # for every epoch the train loss
    test_losses_round = [] # for every epoch the test loss
    
    for e in range(opt.epochs):
        
        train_loss = 0
        test_loss = 0
        batch_amount = 0
        t_batch_amount = 0

        for i, data in enumerate(trainloader):
            input_pc, input_sn, input_seg, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_seg, input_node, input_node_knn_I)
        
            batch_amount += input_seg.size()[0]

            model.optimize()

            train_loss += model.loss_segmenter.cpu().data * input_seg.size()[0]            
            # print(model.test_loss_segmenter.item())
            # print(errors)
            # print()
            
        train_loss /= batch_amount
        
        train_losses_round.append(train_loss.item())
        
        
        for j, t_data in enumerate(testloader):
            t_input_pc, t_input_sn, t_input_seg, t_input_node, t_input_node_knn_I = t_data
            model.set_input(t_input_pc, t_input_sn, t_input_seg, t_input_node, t_input_node_knn_I)
        
            t_batch_amount += t_input_seg.size()[0]
            
            model.test_model()

            test_loss += model.loss_segmenter.cpu().data * t_input_seg.size()[0]            
            # print(model.test_loss_segmenter.item())
            # print(errors)
            # print()
            
        test_loss /= t_batch_amount
        
        test_losses_round.append(test_loss.item())
        
    return train_losses_round, test_losses_round       


if __name__=='__main__':
    # trainset = ArchesLoader(opt.dataroot, 'train', opt)
    # dataset_size = len(trainset)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    # print('#training point clouds = %d' % len(trainset))

    # testset = ArchesLoader(opt.dataroot, 'test', opt)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)


    # visualizer = Visualizer(opt)

    # create model, optionally load pre-trained model

    # TODO: Train on 10% labeled data.
    dataset = ArchesLoader(opt.dataroot, 'all', opt)
    training_size = round(opt.train_frac * len(dataset))
    
    train_losses = []
    test_losses = []
    
    for _ in range(opt.avg_rounds):        
        model = Model(opt)
        if opt.pretrain is not None:
            model.encoder.load_state_dict(torch.load(opt.pretrain))
            
        trainset, validationset = torch.utils.data.random_split(dataset, [training_size, len(dataset) - training_size])

        train_loss_round, test_loss_round = train_model(model, trainset, validationset, opt)
        train_losses.append(train_loss_round)        
        test_losses.append(test_loss_round)
        
        test_model(model, validationset, opt.cluster_save_dir, opt)
        
        # if test_loss_round < best_test_loss:
        #     best_test_loss = test_loss_round
        #     print("Saving network...")
        #     ae_model.save_network(ae_model.encoder, 'encoder', 'boxes_5_5_5_labeled_%depochs' % opt.epochs, opt.gpu_id)
        #     ae_model.save_network(ae_model.decoder, 'decoder', 'boxes_5_5_5_labeled_%depochs' % opt.epochs, opt.gpu_id)

        # Add recorded train and test losses to arrays.
        # train_losses.append(train_losses_round)
        # test_losses.append(test_losses_round)
            
    avg_train_losses = np.average(np.array(train_losses), axis=0) # should be 25 x 1
    avg_test_losses = np.average(np.array(test_losses), axis=0)
    
    plot_train_test_loss(opt.epochs, avg_train_losses, avg_test_losses)
    
    # load pre-trained model
    # folder = 'checkpoints/'
    # model_epoch = '2'
    # model_acc = '0.914946'
    # model.encoder.load_state_dict(torch.load(folder + model_epoch + '_' + model_acc + '_net_encoder.pth'))
    # model.segmenter.load_state_dict(torch.load(folder + model_epoch + '_' + model_acc + '_net_segmenter.pth'))

    # best_iou = 0
    # for epoch in range(opt.epochs):

    #     epoch_iter = 0
    #     for i, data in enumerate(trainloader):
    #         iter_start_time = time.time()
    #         epoch_iter += opt.batch_size

    #         input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
    #         model.set_input(input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I)

    #         model.optimize()

    #         if i % 100 == 0:
    #             # print/plot errors
    #             t = (time.time() - iter_start_time) / opt.batch_size

    #             errors = model.get_current_errors()
                
    #             print(model.test_loss_segmenter.item())
    #             print(errors)
    #             print()

    #             # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    #             # visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

    #             # print(model.autoencoder.encoder.feature)
    #             # visuals = model.get_current_visuals()
    #             # visualizer.display_current_results(visuals, epoch, i)

    #     # test network
    #     if epoch >= 0 and epoch%1==0:
    #         batch_amount = 0
    #         model.test_loss_segmenter.data.zero_()
    #         model.test_accuracy_segmenter.data.zero_()
    #         model.test_iou.data.zero_()
    #         for i, data in enumerate(testloader):
    #             input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
    #             model.set_input(input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I)
    #             model.test_model()

    #             batch_amount += input_label.size()[0]

    #             # # accumulate loss
    #             model.test_loss_segmenter += model.loss_segmenter.detach() * input_label.size()[0]

    #             _, predicted_seg = torch.max(model.score_segmenter.data, dim=1, keepdim=False)
                
    #             print(f"input_pc: {input_pc}")
                
                
    #             print(f"input_sn: {input_sn}")
                
    #             # Predicted_seg == output? --> only labels
    #             print(f"predicted_seg: {predicted_seg}")
                
    #             # Ground truth 
    #             print(f"input_seg: {input_seg}")
                
    #             # Write output to new las file
    #             # Data is array with [x,y,z] arrays inside: [[x1,y1,z1], [x2,y2,z2], ..., [xn,yn,zn]]
	# 	# my_data = np.hstack((my_data_xx.reshape((-1, 1)), my_data_yy.reshape((-1, 1)), my_data_zz.reshape((-1, 1))))
    #             my_data = input_pc.numpy()[0]
	# 	# 1. Create a new header
    #             header = laspy.LasHeader(point_format=6, version="1.4")
	# 	#header.offsets = np.min(my_data, axis=0)

	# 	# 2. Create a Las
    #             las = laspy.LasData(header)

    #             las.x = my_data[0] # Array with all x coefficients. [x1, x2, ..., xn]
    #             las.y = my_data[1]
    #             las.z = my_data[2]
    #             las.classification = predicted_seg.cpu().numpy()[0] # Set labels of every point.

    #             las.write("train_seg_catenary.las")
    #             break
    #             #correct_mask = torch.eq(predicted_seg, model.input_seg).float()
    #             #test_accuracy_segmenter = torch.mean(correct_mask)
    #             #model.test_accuracy_segmenter += test_accuracy_segmenter * input_label.size()[0]

    #             # segmentation iou
    #             test_iou_batch = losses.compute_iou(model.score_segmenter.cpu().data, model.input_seg.cpu().data, model.input_label.cpu().data, visualizer, input_pc.cpu().data)
    #             model.test_iou += test_iou_batch * input_label.size()[0]

    #             # print(test_iou_batch)
    #             # print(model.score_segmenter.size())

    #         print(batch_amount)
    #         model.test_loss_segmenter /= batch_amount
    #         model.test_accuracy_segmenter /= batch_amount
    #         model.test_iou /= batch_amount
    #         if model.test_iou.item() > best_iou:
    #             best_iou = model.test_iou.item()
    #         print('Tested network. So far best segmentation: %f' % (best_iou) )

    #         # # save network
    #         # if model.test_iou.item() > 0.835:
    #         #     print("Saving network...")
    #         #     model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_iou.item()), opt.gpu_id)
    #         #     model.save_network(model.segmenter, 'segmenter', '%d_%f' % (epoch, model.test_iou.item()), opt.gpu_id)

    #     # # learning rate decay
    #     # if epoch%30==0 and epoch>0:
    #     #     model.update_learning_rate(0.5)

    #     # save network
    #     # if epoch%20==0 and epoch>0:
    #     #     print("Saving network...")
    #     #     model.save_network(model.classifier, 'cls', '%d' % epoch, opt.gpu_id)





