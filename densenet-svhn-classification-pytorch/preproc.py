import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import utils

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

datasets_stats={'cifar10':{},'cifar100':{},'svhn':{}}
datasets_stats['cifar10']['mean']=[0.49139968, 0.48215827, 0.44653124]
datasets_stats['cifar10']['std']=[0.24703233, 0.24348505, 0.26158768]
datasets_stats['cifar100']['mean']=[0.5071, 0.4865, 0.4409]
datasets_stats['cifar100']['std']=[0.2673, 0.2564, 0.2762]
datasets_stats['svhn']['mean']=[0.4377, 0.4438, 0.4728]
datasets_stats['svhn']['std']=[0.1980, 0.2010, 0.1970]

def data_transforms(dataset):
    dataset = dataset.lower()
    if dataset == 'cifar10' or dataset == 'cifar100':
        MEAN = datasets_stats[dataset]['mean']
        STD = datasets_stats[dataset]['std']
        transf_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        transf_val = []
    elif dataset == 'svhn':
        MEAN = datasets_stats[dataset]['mean']
        STD = datasets_stats[dataset]['std']
        transf_train = []
        transf_val = []
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf_train = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
        transf_val=[]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf_train = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
        transf_val = []
    #Same preprocessing for ImageNet, Sport8 and MIT67
    elif dataset in utils.LARGE_DATASETS:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        transf_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2)
        ]
        transf_val = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    
    train_transform = transforms.Compose(transf_train + normalize)
    valid_transform = transforms.Compose(transf_val + normalize)  # FIXME validation is not set to square proportions, is this an issue?

    return train_transform, valid_transform
