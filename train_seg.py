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

from models import losses
from models.segmenter import Model
from data.arches_loader import ArchesLoader


def cluster_dataset(model, save_dir):
    dataset = ArchesLoader(opt.dataroot, 'all', opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

    for data in dataloader:
        # Get prediction for this batch
        input_pc, input_sn, input_seg, input_node, input_node_knn_I = data #input_label, 
        model.set_input(input_pc, input_sn, input_seg, input_node, input_node_knn_I) #input_label, 

        # Save all prediction per sample to separate file.
        model.test_model()

        print(model.score_segmenter.data[:2])
        _, predicted_seg = torch.max(model.score_segmenter.data, dim=1, keepdim=False)
        print()



if __name__=='__main__':
    # trainset = ArchesLoader(opt.dataroot, 'train', opt)
    # dataset_size = len(trainset)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    # print('#training point clouds = %d' % len(trainset))

    # testset = ArchesLoader(opt.dataroot, 'test', opt)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)


    # visualizer = Visualizer(opt)

    # create model, optionally load pre-trained model
    model = Model(opt)
    if opt.pretrain is not None:
        model.encoder.load_state_dict(torch.load(opt.pretrain))

    cluster_dataset(model, opt.cluster_save_dir)

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





