"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

def calc_psnr_np(sr, hr, scale):
    """ calculate psnr by numpy

    Params:
    sr : numpy.uint8
        super-resolved image
    hr : numpy.uint8
        high-resolution ground truth
    scale : int
        super-resolution scale
    """
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / 255.
    shave = scale
    if diff.shape[1] > 1:
        convert = np.zeros((1, 3, 1, 1), diff.dtype)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff = diff * (convert) / 256
        diff = diff.sum(axis=1, keepdims=True)

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = np.power(valid, 2).mean()
    return -10 * math.log10(mse)

def calc_psnr(sr, hr, scale):
    """ calculate psnr by torch

    Params:
    sr : torch.float32
        super-resolved image
    hr : torch.float32
        high-resolution ground truth
    scale : int
        super-resolution scale
    """
    with torch.no_grad():
        diff = (sr - hr) / 255.
        shave = scale
        if diff.shape[1] > 1:
            diff *= torch.tensor([65.738, 129.057, 25.064],
                    device=sr.device).view(1, 3, 1, 1) / 256
            diff = diff.sum(dim=1, keepdim=True)
        valid = diff[..., shave:-shave, shave:-shave]
        mse = torch.pow(valid, 2).mean()
        return (-10 * torch.log10(mse)).item()


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss