import time
import copy
import numpy as np
import math
import laspy
import operator
import copy
import itertools

from options_active_seg import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from util.data_structures import Superpoint, Point

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
from sklearn.metrics.pairwise import euclidean_distances

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
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    
    # Set background color for active learning rounds.
    for i in range(7): # number of active learning rounds
        plt.axvspan(30 + i * 15, 30 + (i+1) * 15, facecolor='b' if i % 2 == 0 else 'g', alpha=0.25, zorder=-100) 
        
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
    dataloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=opt.nThreads)

    metric = MulticlassConfusionMatrix(opt.classes)

    for i, data in enumerate(dataloader):
        # Get prediction for this batch
        input_pc, input_sn, input_seg, input_s_labels, input_node, input_node_knn_I, box_idx = data #input_label, 
        model.set_input(input_pc, input_sn, input_seg, input_s_labels, input_node, input_node_knn_I) #input_label, 

        # Save all prediction per sample to separate file.
        model.test_model()

        for j in range(len(input_pc)):
            _, predicted_seg = torch.max(model.score_segmenter.data[j], dim=0, keepdim=False)

            metric.update(predicted_seg.cpu(), input_seg[j])

            # Save 100 samples.
            # if i < 100:
            #     save_to_las(input_pc[j].numpy(), predicted_seg.cpu().numpy(), input_seg[j].numpy(), save_dir, i)

    # Get accuracy and mean IoU for all data.    
    conf_matrix = metric.compute()
    ious = get_iou(conf_matrix)
    print(f"Overall accuracy is {get_overall_acc(conf_matrix * 100)} and the mean IoU is {avg(ious) * 100}")
    
    for class_num, mIoU in enumerate(ious):
        print(f"Class {class_num} had a mIoU of {mIoU * 100}")

        
def train_model(model, trainset, validationset, epochs, opt):    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    testloader = torch.utils.data.DataLoader(validationset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    
    train_losses_round = [] # for every epoch the train loss
    test_losses_round = [] # for every epoch the test loss
    
    for e in range(epochs):
        
        train_loss = 0
        test_loss = 0
        batch_amount = 0
        t_batch_amount = 0

        for i, data in enumerate(trainloader):
            input_pc, input_sn, input_seg, input_s_labels, input_node, input_node_knn_I, box_idx = data
            model.set_input(input_pc, input_sn, input_seg, input_s_labels, input_node, input_node_knn_I)
        
            batch_amount += input_seg.size()[0]

            model.optimize()

            train_loss += model.loss_segmenter.cpu().data * input_seg.size()[0]
            
        train_loss /= batch_amount
        
        train_losses_round.append(train_loss.item())
        
        
        for j, t_data in enumerate(testloader):
            t_input_pc, t_input_sn, t_input_seg, t_input_s_labels, t_input_node, t_input_node_knn_I, box_idx = t_data
            model.set_input(t_input_pc, t_input_sn, t_input_seg, t_input_s_labels, t_input_node, t_input_node_knn_I)
        
            t_batch_amount += t_input_seg.size()[0]
            
            model.test_model()

            test_loss += model.loss_segmenter.cpu().data * t_input_seg.size()[0]
            
        test_loss /= t_batch_amount
        
        test_losses_round.append(test_loss.item())
        
    return train_losses_round, test_losses_round

def get_diversity_score(unlabelled_superpoints, s):
    min_dist = -1
    
    # s_feature = s.get_avg_feature()
    
    for cur_s in unlabelled_superpoints:
        if s.label == cur_s.label: continue
        
        # Calculating every distance twice! Create matrix with distances? -> It is symmetrical and thus only half have to be calculated.
        cur_dist = np.linalg.norm(s.avg_point_feature - cur_s.avg_point_feature)
        
        if min_dist == -1 or cur_dist < min_dist:
            min_dist = cur_dist
    
    return min_dist
    
def get_diversity_score2(s, s_idx, distance_matrix):
    return np.min(distance_matrix[s_idx][np.nonzero(distance_matrix[s_idx])])
    
def calc_class_features(labelled_superpoints, num_classes): # For every class, get avg feature and return as dict
    feature_list = [[] for _ in range(num_classes)] 
    
    for s in labelled_superpoints:
        # Append feature to feature_list[class of s]
        point_f_np = np.asarray([p.feature for p in s.points])
        s_feature = np.average(point_f_np.T, axis=1)
        assert(s_feature.shape[0] == 128)
        feature_list[s.get_class_label()].append(s_feature)
    
    # Return normalized features.
    return {i: (avg(feature_list[i]) / np.linalg.norm(avg(feature_list[i])) if feature_list[i] else [0]) for i in range(num_classes)}

def get_class_diversity_score(class_features, s): # Get class diversity score for a unlabelled superpoint
    # Loop over class features and record min distance of superpoint to class feature.
    min_dist = -1
    
    s_feature = s.avg_point_feature
    
    # If a class does not yet have any labelled points, its length is 1 not considered here.
    for cf in [cf for cf in class_features.values() if len(cf) > 1]:
        cur_dist = np.linalg.norm(s_feature - cf)
        
        if min_dist == -1 or cur_dist < min_dist:
            min_dist = cur_dist
            
    return min_dist
    
def bvsb(p): # compute best-vs-second-best uncertainty for a point
    assert(p.probabilities.shape[0] == 14)
    sorted_ps = np.sort(p.probabilities)
    return 1 - (sorted_ps[-1] - sorted_ps[-2])

def uncertainty(s): # uncertainty for superpixel s -> list with probabilities
    return np.mean(np.vectorize(bvsb)(s.points)) # / s.shape[0]

def penalise_superpoints(superpoints, num_boxes, decay_rate=0.75):
    weights = {i: 1 for i in range(num_boxes)}
    
    for s in superpoints:
        # Weights =< 1 so multiplying decreases informativeness
        s.informativeness *= weights[s.box_index]
        # Decay rate is between 0-1.
        weights[s.box_index] *= decay_rate
        
    return superpoints

def calc_dist(arr):
    if arr[0] != arr[1]:
        # Calc distance and save it unnormalized in [i][j] and [j][i]
        dist = np.linalg.norm(unlabelled_superpoints[i].avg_point_feature - unlabelled_superpoints[j].avg_point_feature)
        distance_matrix[i][j] = dist
        distance_matrix[j][i] = dist
        
def take_point_feature(s):
    return s.avg_point_feature

# Uncertainty based selection works now!!!
def informativeness_selection(superpoint_list, budget, num_boxes):
    # Sort based on informativeness and then just loop over until budget is done.
    chosen_superpoints = {} # dict with superpoint label: class label pairs
    
    unlabelled_superpoints = [s for s in superpoint_list if not s.is_labelled]
    labelled_superpoints = [s for s in superpoint_list if s.is_labelled]
    
    # print(f"The number of unlabelled superpoints is {len(unlabelled_superpoints)} and the number of labelled superpoints is {len(labelled_superpoints)}")
    
    class_features = calc_class_features(labelled_superpoints, 14) # hardcoded num_classes
    assert(len(class_features) == 14)
    
    bt0 = time.time()
    for s in unlabelled_superpoints:
        assert(s.budget == len(s.points))
        s.avg_point_feature = np.average([p.feature for p in s.points], axis=0) / np.linalg.norm(np.average([p.feature for p in s.points], axis=0))
    bt1 = time.time()
    # print(f"Calculating the avg point feature for each superpoint took {bt1 - bt0} seconds.")
    
    ct0 = time.time()
    # Create numpy matrix with distances to superpoints. (Not min distances yet!)
    # TODO: use numpy + map instead of itertools
    # f = np.vectorize(take_point_feature)
    avg_point_features = [s.avg_point_feature for s in unlabelled_superpoints]
    # print(f"Vectorized unlabelled superpoints is {f(unlabelled_superpoints)[:10]}")
    distance_matrix = euclidean_distances(avg_point_features, avg_point_features)
    
    # print(f"Shape of distance matrix is {distance_matrix.shape}")
    # distance_matrix = np.zeros((len(unlabelled_superpoints), len(unlabelled_superpoints)))
    # combinations = np.meshgrid(np.arange(len(unlabelled_superpoints), np.arange(len(unlabelled_superpoints))
    
                                         
    # print(f"The number of pairs to check is {len(itertools.combinations(range(len(unlabelled_superpoints)), 2))}")
    # for i, j in itertools.combinations(range(len(unlabelled_superpoints)), 2):
    #     if i != j:
    #         # Calc distance and save it unnormalized in [i][j] and [j][i]
    #         dist = np.linalg.norm(unlabelled_superpoints[i].avg_point_feature - unlabelled_superpoints[j].avg_point_feature)
    #         distance_matrix[i][j] = dist
    #         distance_matrix[j][i] = dist
            
    ct1 = time.time()
    # print(f"Creating the distance matrix took {ct1 - ct0} seconds.") # ~30 seconds
        
    for i,s in enumerate(unlabelled_superpoints):
        t0 = time.time() # seconds
        s.uncertainty = uncertainty(s)
        t1 = time.time()
        # print(f"Calculating uncertainty took {t1 - t0} seconds.")
        s.class_diversity = get_class_diversity_score(class_features, s)
        t2 = time.time()
        # print(f"Calculating class diversity took {t2 - t1} seconds.")
        
        # TODO: try to speed this up! -> 0.3 seconds for every superpoint == multiple hours to get all!
        # s.diversity = get_diversity_score(unlabelled_superpoints, s)
        s.diversity = get_diversity_score2(s, i, distance_matrix)
        t3 = time.time()
        # print(f"Calculating feature diversity took {t3 - t2} seconds.")
        
        s.update_informativeness(ALPHA, BETA, GAMMA)
        t4 = time.time()
        # print(f"Updating informativeness took {t4 - t3} seconds.")
        # print(f"Updating a single superpoint took {t4 - t0} seconds.")
        # print()
    
    # print("Calculated informativeness for all unlabelled superpoints.")
    
    unlabelled_superpoints.sort(key=operator.attrgetter("informativeness"), reverse=True) # Sort on informativeness
    
    # print("Sorted all unlabelled superpoints by informativeness.")

    # print(f"First ten most informative superpoints are {[[s.informativeness, s.uncertainty, s.diversity, s.class_diversity] for s in superpoint_list[:10]]}")
    # print(f"First ten most informative superpoints with highest class_diversity are {[s.class_diversity for s in unlabelled_superpoints][:10]}")

    # TODO: diversity-aware selection based on box index
    # Penalise superpoints from same box with lower scores.
    decay_rate = 0.75
    unlabelled_superpoints = penalise_superpoints(unlabelled_superpoints, num_boxes, decay_rate)
       
    # print("Penalised all unlabelled superpoints by informativeness.")
    
    # Sort again with added penalisation
    unlabelled_superpoints.sort(key=operator.attrgetter("informativeness"), reverse=True) # Sort on informativeness
    
    # print("Sorted all unlabelled superpoints by adjusted informativeness.")

    # print(f"First ten most informative superpoints are {[[s.informativeness, s.uncertainty, s.diversity, s.class_diversity] for s in superpoint_list][:10]}")
    
    idx = 0
    while budget > 0:
        superpoint = unlabelled_superpoints[idx]
        s_label = superpoint.label
        s_cost = superpoint.budget
                
        # Add supervoxel if budget allows. If budget is under 50, break.
        if budget >= s_cost:
            chosen_superpoints[s_label] = superpoint.get_class_label()
            budget -= s_cost
        elif budget < 50:
            break
            
        idx += 1
            
    return chosen_superpoints  

def random_selection(superpoint_list, budget):
    chosen_superpoints = {} # dict with superpoint label: class label pairs
    
    # print(f"Superpoint_list has a length of {len(superpoint_list)}.")
    # print(f"The point budget is {budget}.")

    unlabelled_superpoints = [s for s in superpoint_list if not s.is_labelled]
    labelled_superpoints = [s for s in superpoint_list if s.is_labelled]
    
    # print(f"The number of unlabelled superpoints is {len(unlabelled_superpoints)} and the number of labelled superpoints is {len(labelled_superpoints)}")
    
    # Sort randomly and then just loop over until budget is done.
    random.shuffle(unlabelled_superpoints)
    
    idx = 0
    while budget > 0:
        if idx == len(unlabelled_superpoints):
            print(f"Something went wrong! idx == len(unlabelled_superpoints) == {idx}")
            break
            
        superpoint = unlabelled_superpoints[idx]
        
        # if idx < 10:
        #     print(f"Current superpoints has budget of {superpoint.budget} and points {superpoint.points}")
            
        s_label = superpoint.label
        s_cost = len(superpoint.points)
        
        assert(s_cost == len(superpoint.points))
        
        # Add supervoxel if budget allows. If budget is under 50, break.
        if budget >= s_cost:
            chosen_superpoints[s_label] = superpoint.get_class_label()
            budget -= s_cost
        elif budget < 50:
            break
            
        idx += 1
            
    return chosen_superpoints  

def get_index(l, s_label):
    for i, el in enumerate(l):
        if el.label == s_label:
            return i
        
    return -1

def al_round(model, trainset, labelled_superpoints, budget, opt, pseudo_labelling=False):
    # Get list of all supervoxel labels + also the ones in labelled set!!!.
    # unlabelled_supervoxel_counts = [] # (label, count), count needed for annotation budget
    
    # unlabelled_supervoxel_counts = (label, count for every unlabelled superpoint)
    # labelled_supervoxels == global label
    
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=opt.nThreads)
    
    # TODO: check difference between get_all_supervoxels() (> 80.000 superpoints) and superpoint_list (>50.000 superpoints).
    # get_all_superpoints() uses the non-downsamples inputs!
    # print(f"The length of the trainset dataloader is {len(dataloader)}")
    
    superpoint_list = []    
    s_labels_in_list = []    
    pseudo_superpoints = {}
    # p_count = 0
    
    for idx, item in enumerate(dataloader): # A single box consisting of multiple supervoxels
        # GPU does not release memory after one testing?!?!?!
        input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I, box_idx = item
        model.set_input(input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I) #input_label, 
        
        # print(f"Shape of input point cloud is {input_pc.shape}")
        # after_loading_input_mem = torch.cuda.memory_allocated(opt.device)
        # print(f"Amount of GPU memory allocated in MB after loading input: {after_loading_input_mem / 1000000}")

        model.test_model()
        
        # model.score_segmenter == B x 14 x 1024
        # model.point_features == B x 128 x 1024
        # predictions == B x 14 x 1024
        
        # predictions = F.softmax(model.score_segmenter.data, dim=2) # B x 14 x 1024
        # print(f"Shape of predictions is {predictions.shape}")
        # print(f"Shape of predictions[0] is {predictions[0].shape}")
        
        for i in range(input_pc.shape[0]):
            for j in range(input_pc[i].shape[1]): # loop over points
                # p_count += 1
                
                # Get probabilities
                probabilities = F.softmax(model.score_segmenter.data[i].T, dim=1) # 1024 x 14                    
                
                # Make point structure 
                assert(model.point_features[i].T[j].cpu().detach().numpy().shape[0] == 128)
                
                cur_point = Point(input_label[i][j].cpu().detach().item(), model.point_features[i].T[j].cpu().detach().numpy(), probabilities[j].cpu().detach().numpy())

                # Add point to superpoint
                cur_s_label = input_s_labels[i][j].cpu().detach().item()            
                
                if cur_s_label not in s_labels_in_list: # Make new superpoints with cur_s_label
                    superpoint_list.append(Superpoint(cur_s_label, box_idx.cpu().detach().item(), [cur_point], is_labelled=cur_s_label in labelled_superpoints))
                    s_labels_in_list.append(cur_s_label)
                else: # Add point to existing superpoint
                    s_index = s_labels_in_list.index(cur_s_label)
                    superpoint_list[s_index].add_point(cur_point)
                                
        
        if pseudo_labelling:
            for s in [sp for sp in superpoint_list if not sp.is_labelled]:      
                # Pseudo-label if p>95% and pseudo-labelling is turned on. Get average probability of all points
                highest_ps = [np.max(p.probabilities) for p in s]
                    
                avg_high_p = np.mean(highest_ps)
                
                if avg_high_p > 0.95:
                    # Add label
                    pseudo_superpoints[s.label] = s.get_class_label()
                    
                    # Mark as labelled
                    s.is_labelled = True
                    
            # for j in range(len(input_pc)): # Do this for whole segmenter data in one go instead per point.
            #     prediction = F.softmax(model.score_segmenter.data[j], dim=1) #logits -> transform to probabilities

            # print(f"prediction has the shape of {prediction.shape} (should be B x 1024 x 14). Length of 1 prediction is {prediction.shape[0]}.")

            # print(f"Shape of np.isin() is {np.isin(input_s_labels[0], labelled_supervoxels, invert=True).shape}")
            # print(f"Shape of input_s_label is {input_s_labels.shape}.")
            # print(f"Length of labelled_supervoxels is {len(labelled_supervoxels)}")

            # We do not need this for informativeness selector.
#             # Get unique s_labels
            # unlabelled_supervoxels = input_s_labels[i][np.isin(input_s_labels[i], labelled_superpoints, invert=True)] #[l for l in input_s_labels if l not in labelled_supervoxels]
            # unique_s_labels, counts = np.unique(unlabelled_supervoxels, return_counts=True)        

#             # Add [label, count] arrays to the unlabelled array.
#             unlabelled_supervoxel_counts.extend(np.squeeze(np.dstack((unique_s_labels, counts))))
    
    # print(f"Unlabelled supervoxels counts is {unlabelled_supervoxel_counts[:3]}")
    # print(f"We looped over {p_count} points to add them to superpoints!")
    # print(f"Superpoint_list has a length of {len(superpoint_list)}. First two superpoints are {superpoint_list[:2]}")
    
    assert(sum([x.budget for x in superpoint_list]) == 425984)

    # Determine best supervoxels. (Also exclude the pseudolabelled superpoints!)
    most_informative_superpoints = random_selection(superpoint_list, budget)
    # most_informative_superpoints = informativeness_selection(superpoint_list, budget, math.ceil(len(dataloader.dataset) * 1.25)) #unlabelled_supervoxel_counts, budget)
    
    # s_dict = {s_label:  for s_label in most_informative_superpoints}
    
    if pseudo_labelling:
        most_informative_superpoints.update(pseudo_superpoints)
    
    return most_informative_superpoints # dict with {s_label: class_label} pairs

ALPHA = 0.75
BETA = 0.25
GAMMA = 0.75

# TODO: use dict with maps superpoint label to class label??
if __name__=='__main__':
    dataset = ArchesLoader(opt.dataroot, 'all', opt)
    # all_supervoxel_labels = get_all_supervoxels(dataset.dataset)
    
    labelled_superpoints = {}
    
    # Split into 80% training and 20% validation
    training_size = round(opt.train_frac * len(dataset))
    test_size = len(dataset) - training_size
    first_generator = torch.Generator().manual_seed(42)
    
    train_dataset, validationset = torch.utils.data.random_split(dataset, [training_size, test_size], generator=first_generator)
    
    testloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=opt.nThreads)
    
    # Something like 5% but if 15% use same data
    five_percent = round(0.05 * len(dataset))
    unlabelled_size = len(train_dataset) - (3 * five_percent)
    second_generator = torch.Generator().manual_seed(42)
    five1, five2, five3, unlabelled_set = torch.utils.data.random_split(train_dataset, [five_percent, five_percent, five_percent, unlabelled_size], generator=first_generator)
    
    # Check if randomness is fixed here. It is fixed!
#     c = 0
#     for elem in validationset:
#         input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I = elem

#         print(f"First five points of first 5 samples validationset are {input_pc[:, :5]}")
#         c += 1
#         if c == 5: break
        
#     c = 0
#     for elem in unlabelled_set:
#         input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I = elem

#         print(f"First five points of first 5 samples unlabelled_set are {input_pc[:, :5]}")
#         c += 1
#         if c == 5: break
    
#     c = 0
#     for elem in five1:
#         input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I = elem

#         print(f"First five points of first 5 samples 1st five percent are {input_pc[:, :5]}")
#         c += 1
#         if c == 5: break
        
#     c = 0
#     for elem in five2:
#         input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I = elem

#         print(f"First five points of first 5 samples 2nd five percent are {input_pc[:, :5]}")
#         c += 1
#         if c == 5: break
        
#     c = 0
#     for elem in five3:
#         input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I = elem

#         print(f"First five points of first 5 samples 3rd five percent are {input_pc[:, :5]}")
#         c += 1
#         if c == 5: break
    
    if opt.init_train_frac == 0.05:
        labelled_trainset = five1
    elif opt.init_train_frac == 0.1:
        labelled_trainset = torch.utils.data.ConcatDataset([five1, five2])
    elif opt.init_train_frac == 0.15:
        labelled_trainset = torch.utils.data.ConcatDataset([five1, five2, five3])
    else:
        raise Exception('Expected a initial train fraction of 0.05, 0.1 or 0.15!') # Don't! If you catch, likely to hide bugs.
        
    for item in labelled_trainset:
        input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I, box_idx = item
        
        # print(f"Shape of input_sn is {input_label.shape} and shape of input_s_labels is {input_s_labels.shape}")
        
        # Make key-value pairs of superpoint_label: class_label
        d_lists = {}
        for i in range(input_s_labels.shape[0]):
            s_label = input_s_labels[i].cpu().detach().item()
            c_label = input_label[i].cpu().detach().item()
            
            if s_label in d_lists:
                d_lists[s_label].append(c_label)
            else:
                d_lists[s_label] = [c_label]
            
        d_labels = {k: max(v, key=v.count) for k, v in d_lists.items()} 
        
        labelled_superpoints.update(d_labels)
        # labelled_superpoints.extend(np.unique(input_s_labels))
        
    print(f"Number of initially labelled superpoints is {len(labelled_superpoints)}")

    # print(f"First 10 labelled superpoints are {list(labelled_superpoints.items())[:10]}")
        
    train_losses = []
    test_losses = []
    
    for i in range(opt.avg_rounds):
        print(f"\nAveraging round {i}!")

        # Init train and test loss for this round
        train_loss_round = []
        test_loss_round = []
        
        # Pre-train model on initial training data
        model = Model(opt, copy.deepcopy(labelled_superpoints))
        
        if opt.init_train_frac == 0.05:
            assert(len(model.labelled_superpoints) == 3630)
        elif opt.init_train_frac == 0.1:
            assert(len(model.labelled_superpoints) == 7558)
        elif opt.init_train_frac == 0.15:
            assert(len(model.labelled_superpoints) == 11102)
        
        s_labels = []
        for item in labelled_trainset:
            input_pc, input_sn, input_label, input_s_labels, input_node, input_node_knn_I, box_idx = item
            
            for s_label in input_s_labels:
                if s_label not in s_labels:
                    s_labels.append(s_label)
                
        assert(len(labelled_superpoints) == len(s_labels))
            
        # before_train_mem = torch.cuda.memory_allocated(opt.device)
        # print(f"Amount of GPU memory allocated in MB before training (approx. 15.000 available): {before_train_mem / 1000000}")
        train_loss_init, test_loss_init = train_model(model, labelled_trainset, validationset, opt.epochs, opt)

        train_loss_round.extend(train_loss_init)        
        test_loss_round.extend(test_loss_init)
            
        # Trainset is all training boxes. Use labelled_supervoxels array to determine loss.
        # trainset = torch.utils.data.ConcatDataset([five1, five2, five3, unlabelled_set])
        # trainset = train_dataset
        
        # Start active learning rounds
        for j in range(opt.active_rounds):
            # Get new labelled data
            print(f"Active learning round {j}!")
            point_budget = opt.annotation_budget * opt.input_pc_num * len(dataset)
            
            # Get most informative supervoxels (dict with s_label: class_label pairs)
            new_s_points = al_round(model, train_dataset, model.labelled_superpoints, point_budget, opt) # new data is array supervoxel labels
            
            # duplicates = set([x for x in model.labelled_supervoxels if model.labelled_supervoxels.count(x) > 1])
            
            assert(len(model.labelled_superpoints) == len(set(model.labelled_superpoints)))
            
            # Mark new supervoxels as labelled.
            model.add_labelled_superpoints(new_s_points)
            
            assert(len(model.labelled_superpoints) == len(set(model.labelled_superpoints)))
            
            # Train with new labelled supervoxels.
            train_loss, test_loss = train_model(model, train_dataset, validationset, opt.active_epochs, opt)
            train_loss_round.extend(train_loss)        
            test_loss_round.extend(test_loss)
            
        # Test model 
        test_model(model, validationset, opt.cluster_save_dir, opt)
            
        # Add train and test loss of this round to the train and test loss arrays.
        train_losses.append(train_loss_round)        
        test_losses.append(test_loss_round)        
        
            
    avg_train_losses = np.average(np.array(train_losses), axis=0) # should be 25 x 1
    avg_test_losses = np.average(np.array(test_losses), axis=0)
    
    plot_train_test_loss(opt.epochs + (opt.active_rounds * opt.active_epochs), avg_train_losses, avg_test_losses)





