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
import faiss


def random_noise(data, min, max):
    """ Add random noise to each point in the point cloud to augument the dataset
        Input:
          Nx3 array, original point clouds + min and max noise value
        Return:
          Nx3 array, rotated point clouds
    """
    # rand returns floating-point samples from the uniform distribution.
    random_nums = np.random.rand(data.shape[0], data.shape[1]) * (max - min) 
    noise = random_nums + min
    return data + noise


def random_gaussian_noise(data, mean, std):
    """ Add random Gaussian noise to each point in the point cloud to augument the dataset
        Input:
          Nx3 array, original point clouds + mean and std of noise
        Return:
          Nx3 array, rotated point clouds
    """
    # noise = np.random.normal(mean, std, len(data))
    # randn returns floating-point samples from the Gaussian distribution.
    noise = np.clip(std * np.random.randn(data.shape[0], data.shape[1]) + mean, -0.05, 0.05)
    return data + noise


def rotate_point_cloud(data, angle):
    """ Rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds + angle to rotate over in degrees
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = angle * (np.pi / 180.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_90(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.randint(low=0, high=4) * (np.pi/2.0) # 0-3 -> 0pi (0 degrees), 0.5pi (90 degrees), 1pi (180 degrees), 1.5pi (270 degrees)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def random_rotate_point_cloud(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(data.shape, dtype=np.float32)

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(data.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def rotate_point_cloud_with_normal_som(pc, surface_normal, som):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    rotation_angle = np.random.uniform() * 2 * np.pi
    # rotation_angle = np.random.randint(low=0, high=12) * (2*np.pi / 12.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    rotated_pc = np.dot(pc, rotation_matrix)
    rotated_surface_normal = np.dot(surface_normal, rotation_matrix)
    rotated_som = np.dot(som, rotation_matrix)

    return rotated_pc, rotated_surface_normal, rotated_som


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_perturbation_point_cloud_with_normal_som(pc, surface_normal, som, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))

    rotated_pc = np.dot(pc, R)
    rotated_surface_normal = np.dot(surface_normal, R)
    rotated_som = np.dot(som, R)

    return rotated_pc, rotated_surface_normal, rotated_som


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data
