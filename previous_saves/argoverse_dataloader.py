# argoverse_dataloader.py
import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

# Adjust path to SFA root
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.argoverse_dataset import ArgoverseDataset
# from data_process.transformation import OneOf, Random_Rotation, Random_Scaling # If you want data augmentation


def create_train_dataloader(configs):
    """Create dataloader for training on Argoverse"""
    # You would define Argoverse-specific augmentations here.
    # For now, keeping it simple.
    train_lidar_aug = None # Example: OneOf([Random_Rotation(...), Random_Scaling(...)])
    train_dataset = ArgoverseDataset(configs, mode='train', lidar_aug=train_lidar_aug, hflip_prob=configs.hflip_prob,
                                     num_samples=configs.num_samples, target_camera='ring_front_center') # Specify target camera
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)
    return train_dataloader, train_sampler


def create_val_dataloader(configs):
    """Create dataloader for validation on Argoverse"""
    val_sampler = None
    val_dataset = ArgoverseDataset(configs, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples,
                                   target_camera='ring_front_center')
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)
    return val_dataloader


def create_test_dataloader(configs):
    """Create dataloader for testing phase on Argoverse"""
    test_dataset = ArgoverseDataset(configs, mode='test', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples,
                                    target_camera='ring_front_center') # Using front center for test
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)
    return test_dataloader