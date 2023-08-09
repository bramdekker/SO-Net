import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from collections import OrderedDict
import os
import os.path
import json

from . import networks
from . import losses

class Model():
    def __init__(self, opt, labelled_superpoints={}):
        self.opt = opt

        self.encoder = networks.Encoder(opt)
        self.segmenter = networks.Segmenter(opt)

        self.softmax_segmenter = losses.CrossEntropyLossSeg()
        if self.opt.gpu_id >= 0:
            self.encoder = self.encoder.to(self.opt.device)
            self.segmenter = self.segmenter.to(self.opt.device)
            self.softmax_segmenter = self.softmax_segmenter.to(self.opt.device)

        # learning rate_control
        if self.opt.pretrain is not None:
            self.old_lr_encoder = self.opt.lr * self.opt.pretrain_lr_ratio
        else:
            self.old_lr_encoder = self.opt.lr
        self.old_lr_segmenter = self.opt.lr

        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=self.old_lr_encoder,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=0)
        self.optimizer_segmenter = torch.optim.Adam(self.segmenter.parameters(),
                                                    lr=self.old_lr_segmenter,
                                                    betas=(0.9, 0.999),
                                                    weight_decay=0)
        
        # Dict (s_label: class_label).
        self.labelled_superpoints = labelled_superpoints

        # place holder for GPU tensors
        self.input_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        # self.input_label = torch.LongTensor(self.opt.batch_size).fill_(1)
        self.input_seg = torch.LongTensor(self.opt.batch_size, self.opt.classes).fill_(1)
        self.input_s_labels = torch.LongTensor(self.opt.batch_size, self.opt.classes).fill_(1)
        self.input_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.input_node_knn_I = torch.LongTensor(self.opt.batch_size, self.opt.node_num, self.opt.som_k)

        # record the test loss and accuracy
        self.test_loss_segmenter = torch.FloatTensor([0])
        self.test_accuracy_segmenter = torch.FloatTensor([0])
        self.test_iou = torch.FloatTensor([0])

        if self.opt.gpu_id >= 0:
            self.input_pc = self.input_pc.to(self.opt.device)
            self.input_sn = self.input_sn.to(self.opt.device)
            # self.input_label = self.input_label.to(self.opt.device)
            self.input_seg = self.input_seg.to(self.opt.device)
            self.input_s_labels = self.input_s_labels.to(self.opt.device)
            self.input_node = self.input_node.to(self.opt.device)
            self.input_node_knn_I = self.input_node_knn_I.to(self.opt.device)
            self.test_loss_segmenter = self.test_loss_segmenter.to(self.opt.device)
            self.test_accuracy_segmenter = self.test_accuracy_segmenter.to(self.opt.device)
            
    def add_labelled_superpoints(self, s_labels):
        self.labelled_superpoints.update(s_labels)
        # self.labelled_supervoxels.extend(s_labels)

    def set_input(self, input_pc, input_sn, input_seg, input_s_labels, input_node, input_node_knn_I): # input_label, 
        assert(np.logical_and(np.all(input_seg.cpu().numpy() >= 0), np.all(input_seg.cpu().numpy() < self.opt.classes)))

        self.input_pc.resize_(input_pc.size()).copy_(input_pc)
        self.input_sn.resize_(input_sn.size()).copy_(input_sn)
        # self.input_label.resize_(input_label.size()).copy_(input_label)
        self.input_seg.resize_(input_seg.size()).copy_(input_seg)
        self.input_s_labels.resize_(input_s_labels.size()).copy_(input_s_labels)


        assert(np.logical_and(np.all(self.input_seg.cpu().numpy() >= 0), np.all(self.input_seg.cpu().numpy() < self.opt.classes)))

        self.input_node.resize_(input_node.size()).copy_(input_node)
        self.input_node_knn_I.resize_(input_node_knn_I.size()).copy_(input_node_knn_I)
        self.pc = self.input_pc.detach()
        self.sn = self.input_sn.detach()
        self.seg = self.input_seg.detach()
        self.s_labels = self.input_s_labels.detach()


        # self.seg needs to be onehot encoded? -> 4 == [0, 0, 0, 0, 1] etc?
        # # Remove one dimension.
        # self.seg = torch.squeeze(self.seg)

        # print(f"Number of nonzero elements is {torch.count_nonzero(self.seg.data)}")
        # print(f"The dtype of self.seg is {self.seg.dtype}")
        # print(f"Shape of self.seg is {self.seg.shape}. First 2 entries of self.seg are {self.seg.data[:2]}.")

        assert(np.logical_and(np.all(self.seg.data.cpu().numpy() >= 0), np.all(self.seg.data.cpu().numpy() < self.opt.classes)))
        # self.label = self.input_label.detach()


    def forward(self, is_train=False, epoch=None):
        # ------------------------------------------------------------------
        self.point_features = None
        self.feature = self.encoder(self.pc, self.sn, self.input_node, self.input_node_knn_I, is_train, epoch)

        batch_size = self.feature.size()[0]
        feature_num = self.feature.size()[1]
        N = self.pc.size()[2]

        # ------------------------------------------------------------------
        k = self.opt.k
        # BxkNxnode_num -> BxkN, tensor
        _, mask_max_idx = torch.max(self.encoder.mask, dim=2, keepdim=False)  # BxkN
        mask_max_idx = mask_max_idx.unsqueeze(1)  # Bx1xkN
        mask_max_idx_384 = mask_max_idx.expand(batch_size, 384, k*N).detach()
        mask_max_idx_512 = mask_max_idx.expand(batch_size, 512, k*N).detach()
        mask_max_idx_fn = mask_max_idx.expand(batch_size, feature_num, k * N).detach()

        feature_max_first_pn_out = torch.gather(self.encoder.first_pn_out_masked_max , dim=2, index=mask_max_idx_384)  # Bx384xnode_num -> Bx384xkN
        feature_max_knn_feature_1 = torch.gather(self.encoder.knn_feature_1, dim=2, index=mask_max_idx_512)  # Bx512xnode_num -> Bx512xkN
        feature_max_final_pn_out = torch.gather(self.encoder.final_pn_out, dim=2, index=mask_max_idx_fn)  # Bx1024xnode_num -> Bx1024xkN

        self.score_segmenter = self.segmenter(self.encoder.x_decentered,
                                              self.pc,
                                              self.encoder.centers,
                                              self.sn,
                                              self.encoder.first_pn_out,
                                              feature_max_first_pn_out,
                                              feature_max_knn_feature_1,
                                              feature_max_final_pn_out,
                                              self.feature, is_train) # self.input_label,
        
        if not is_train:
            self.point_features = self.segmenter.final_point_features

    def optimize(self, epoch=None):
        self.encoder.train()
        self.segmenter.train()
        self.forward(is_train=True, epoch=epoch)

        self.encoder.zero_grad()
        self.segmenter.zero_grad()

        # TODO: Only add self.seg if supervoxel is activated!
        # self.s_labels == Tensor (Batch, 1024)
        # self.labelled_supervoxels == list
        
        # Goal: get self.seg and self.score_segmenter only for points which supervoxel label is in labelled_supervoxels
        
        # Get id of the labelled points.
        s_labels_np = self.s_labels.detach().cpu().numpy()
        score_segmenter_np = self.score_segmenter.detach().cpu().numpy()
        seg_np = self.seg.detach().cpu().numpy()
        
        labelled_score_segmenter = []
        labelled_seg = []
        
        # Loop over samples in batch:
        for i in range(s_labels_np.shape[0]):
            indices = np.nonzero(np.isin(s_labels_np[i], list(self.labelled_superpoints.keys())))
            
            # Possible that indices is empty.
            # print(indices)
            if indices[0].shape[0] > 0:
                # print(f"Shape of score_segmenter_np[i][:, indices] is {np.squeeze(score_segmenter_np[i][:, indices]).shape} (14x1024)") # (14, 1024)
                # print(np.squeeze(score_segmenter_np[i][:, indices]))
                if np.squeeze(score_segmenter_np[i][:, indices]).ndim == 1: # Just 1 point in segment.
                    labelled_score_segmenter.append(np.squeeze(score_segmenter_np[i][:, indices]).copy()) # Array of arrays with 14 length (probabilities for each point)
                else:
                    for arr in np.squeeze(score_segmenter_np[i][:, indices]).T:
                        labelled_score_segmenter.append(arr.copy()) # Array of arrays with 14 length (probabilities for each point)
                    # labelled_score_segmenter.extend(np.squeeze(score_segmenter_np[i][:, indices]))
                
                # print(f"Shape of seg_np[i][indices] is {seg_np[i][indices].shape} (1024,)")
                
                # TODO: get superpoint labels instead of ground-truth point labels
                cur_point_labels = np.vectorize(self.labelled_superpoints.get)(s_labels_np[i][indices])
                # print(f"The superpoint labels are {cur_point_labels}")
                # print(f"The superpoint label is {max(v, key=v.count)}")
                labelled_seg.extend(cur_point_labels.flatten())
                # labelled_seg.extend(seg_np[i][indices].flatten()) # Labels for each point.
                        

        # Only calc loss if there are some labelled points in the batch.
        if len(labelled_score_segmenter) > 0:
            # print(f"First items in labelled_score_segmenter (length {len(labelled_score_segmenter)}, length of item in list is {labelled_score_segmenter[0].shape[0]}) are {labelled_score_segmenter[:2]}")# {np.asarray(labelled_score_segmenter, dtype=object).shape}")
            # print(f"First items in labelled_seg (length {len(labelled_seg)}) are {labelled_seg[:2]}")# {np.squeeze(np.asarray(labelled_seg)).shape}")
                
        
            labelled_score_segmenter_tensor = torch.tensor(np.asarray([labelled_score_segmenter]), requires_grad=True) # 1x14xN
            labelled_seg_tensor = torch.tensor(np.asarray([labelled_seg])) # 1xN
            self.loss_segmenter = self.softmax_segmenter(self.score_segmenter, self.seg)


        # print(f"First items in labelled_score_segmenter (length {len(labelled_score_segmenter)}) are {labelled_score_segmenter[:2]}")# {np.asarray(labelled_score_segmenter, dtype=object).shape}")
        # print(f"First items in labelled_seg (length {len(labelled_seg)}) are {labelled_seg[:2]}")# {np.squeeze(np.asarray(labelled_seg)).shape}")
        
        # TODO: calc loss per sample, then take mean (or look at how loss is calculated for batches
#         losses = []
#         for s_idx in range(len(labelled_seg)):
#             score_segmenter_tensor = torch.tensor(np.asarray([labelled_score_segmenter[s_idx]]), requires_grad=True) # 1x14xN
#             seg_tensor = torch.tensor(np.asarray([labelled_seg])) # 1xN
            
#             print(f"score_segmenter shape is {score_segmenter.shape} (1x14xN)")
#             print(f"seg_tensor shape is {seg_tensor.shape} (1xN)")
            
#             loss = self.softmax_segmenter(score_segmenter_tensor, seg_tensor)
#             losses.append(loss)
            
        
        # print(np.asarray(labelled_seg).shape)
        # BxN --> N changes deending on how many supervoxels (and thus points) are labelled
        # self.seg = torch.tensor(np.asarray(labelled_seg)).astype(np.int64)
                
        # BxclassxN, where B then number of samples containing labeled supervoxels.
        # self.score_segmenter = torch.tensor(np.squeeze(np.asarray(labelled_score_segmenter, dtype=object)).astype(np.float32), requires_grad=True)
        
        # self.loss_segmenter = sum(l) / len(l)
        
        # self.loss_segmenter = self.softmax_segmenter(self.score_segmenter, self.seg)
        
            # print(f"Output of loss_segmenter is {self.loss_segmenter}")
            # self.loss_segmenter = self.softmax_segmenter(labelled_score_segmenter_tensor, labelled_seg_tensor)
            self.loss_segmenter.backward()

            self.optimizer_encoder.step()
            self.optimizer_segmenter.step()

    def test_model(self): # Here, all are labelled!!!
        self.encoder.eval()
        self.segmenter.eval()
        self.forward(is_train=False)

        self.loss_segmenter = self.softmax_segmenter(self.score_segmenter, self.seg)
        self.loss = self.loss_segmenter
        # print(f"Param inputs for softmax_segmenter has shape {self.score_segmenter.shape}")
        # print(f"Param targets for softmax_segmenter has shape {self.seg.shape}")

        # self.loss_classifier = self.softmax_classifier(self.score_classifier, self.label)

        # :param inputs: BxclassxN == 8 x 14 x 8192
        # :param targets: BxN == 8 x 8192
        
#         s_labels_np = self.s_labels.detach().cpu().numpy()
#         score_segmenter_np = self.score_segmenter.detach().cpu().numpy()
#         seg_np = self.seg.detach().cpu().numpy()
        
#         labelled_score_segmenter = []
#         labelled_seg = []
        
#         # Loop over samples in batch:
#         for i in range(s_labels_np.shape[0]):
#             indices = np.nonzero(np.isin(s_labels_np[i], self.labelled_supervoxels))
            
#             # Possible that indices is empty.
#             # print(indices)
#             if indices[0].shape[0] > 0:
#                 # print(f"Shape of score_segmenter_np[i][:, indices] is {np.squeeze(score_segmenter_np[i][:, indices]).shape} (14x1024)") # (14, 1024)
#                 # print(np.squeeze(score_segmenter_np[i][:, indices]))
#                 if np.squeeze(score_segmenter_np[i][:, indices]).ndim == 1: # Just 1 point in segment.
#                     labelled_score_segmenter.append(np.squeeze(score_segmenter_np[i][:, indices]).copy()) # Array of arrays with 14 length (probabilities for each point)
#                 else:
#                     for arr in np.squeeze(score_segmenter_np[i][:, indices]).T:
#                         labelled_score_segmenter.append(arr.copy()) # Array of arrays with 14 length (probabilities for each point)
#                     # labelled_score_segmenter.extend(np.squeeze(score_segmenter_np[i][:, indices]))
                
#                 # print(f"Shape of seg_np[i][indices] is {seg_np[i][indices].shape} (1024,)")
#                 labelled_seg.extend(seg_np[i][indices].flatten()) # Labels for each point.
                        

#         # Only calc loss if there are some labelled points in the batch.
#         if len(labelled_score_segmenter) > 0:
#             # print(f"First items in labelled_score_segmenter (length {len(labelled_score_segmenter)}, length of item in list is {labelled_score_segmenter[0].shape[0]}) are {labelled_score_segmenter[:2]}")# {np.asarray(labelled_score_segmenter, dtype=object).shape}")
#             # print(f"First items in labelled_seg (length {len(labelled_seg)}) are {labelled_seg[:2]}")# {np.squeeze(np.asarray(labelled_seg)).shape}")
                
        
#             labelled_score_segmenter_tensor = torch.tensor(np.asarray([labelled_score_segmenter]), requires_grad=True) # 1x14xN
#             labelled_seg_tensor = torch.tensor(np.asarray([labelled_seg])) # 1xN
#             self.loss_segmenter = self.softmax_segmenter(self.score_segmenter, self.seg)

        
#         # Loop over samples in batch:
#         for i in range(s_labels_np.shape[0]):
#             indices = np.nonzero(np.isin(s_labels_np[i], self.labelled_supervoxels))
            
#             labelled_score_segmenter.append(score_segmenter_np[i][:, indices])
#             labelled_seg.append(seg_np[i][indices])
        
        # self.score_segmenter = torch.tensor(np.squeeze(np.asarray(labelled_score_segmenter, dtype=object)).astype(np.float32), requires_grad=True)
            # self.seg = torch.tensor(np.squeeze(np.asarray(labelled_seg, dtype=object)).astype(np.int64))

            # self.loss_segmenter = self.softmax_segmenter(self.score_segmenter, self.seg)
            # self.loss_segmenter = self.softmax_segmenter(self.score_segmenter[indices], self.seg[indices])
            # self.loss = self.loss_segmenter

        # visualization with visdom
    def get_current_visuals(self):
        # display only one instance of pc/img
        input_pc_np = self.input_pc[0].cpu().numpy().transpose() # Nx3
        pc_color_np = np.zeros(input_pc_np.shape)  # Nx3
        gt_pc_color_np = np.zeros(input_pc_np.shape)  # Nx3

        # construct color map
        _, predicted_seg = torch.max(self.score_segmenter.data[0], dim=0, keepdim=False)  # 50xN -> N
        predicted_seg_np = predicted_seg.cpu().numpy()  # N
        gt_seg_np = self.seg.data[0].cpu().numpy()  # N

        color_map_file = os.path.join(self.opt.dataroot, 'part_color_mapping.json')
        color_map = json.load(open(color_map_file, 'r'))
        color_map_np = np.rint((np.asarray(color_map)*255).astype(np.int32))  # 50x3

        for i in range(input_pc_np.shape[0]):
            pc_color_np[i] = color_map_np[predicted_seg_np[i]]
            gt_pc_color_np[i] = color_map_np[gt_seg_np[i]]

        return OrderedDict([('pc_colored_predicted', [input_pc_np, pc_color_np]),
                            ('pc_colored_gt',        [input_pc_np, gt_pc_color_np])])

    def get_current_errors(self):
        # self.score_segmenter: BxclassesxN
        _, predicted_seg = torch.max(self.score_segmenter.data, dim=1, keepdim=False)
        correct_mask = torch.eq(predicted_seg, self.input_seg).float()
        train_accuracy_segmenter = torch.mean(correct_mask)

        return OrderedDict([
            ('train_loss_seg', self.loss_segmenter.item()),
            ('train_accuracy_seg', train_accuracy_segmenter),
            ('test_loss_seg', self.test_loss_segmenter.item()),
            ('test_acc_seg', self.test_accuracy_segmenter.item()),
            ('test_iou', self.test_iou.item())
        ])

    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id >= 0 and torch.cuda.is_available():
            # torch.cuda.device(gpu_id)
            network.to(self.opt.device)

    def update_learning_rate(self, ratio):
        # encoder
        lr_encoder = self.old_lr_encoder * ratio
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr_encoder
        print('update encoder learning rate: %f -> %f' % (self.old_lr_encoder, lr_encoder))
        self.old_lr_encoder = lr_encoder

        # segmentation
        lr_segmenter = self.old_lr_segmenter * ratio
        for param_group in self.optimizer_segmenter.param_groups:
            param_group['lr'] = lr_segmenter
        print('update segmenter learning rate: %f -> %f' % (self.old_lr_segmenter, lr_segmenter))
        self.old_lr_segmenter = lr_segmenter
