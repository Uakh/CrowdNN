# -*- coding: utf-8 -*-
import re
import functools
import numpy as np
from os import path
from skimage import io, util, filters
from scipy import ndimage, sparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.utils.data
import statistics
import itertools

def parse_db(file_path):
    """
    Parses a DB text file
    """
    pattern = re.compile('\.tiff$')
    imgs = []
    cf = ''
    with open(file_path, 'r') as f:
        for line in f:
            if pattern.search(line):
                cf = path.dirname(file_path) + '/Data/' + line.rstrip('\n')
                image = io.imread(cf)
                imgs.append({'path' : path.abspath(cf),
                             'image' : image,
                             'pos' : [],
                             'dens' : np.zeros(image.shape, np.float32)})
            else:
                item = line.rstrip('\n').split()
                item = {'x' : int(item[1]),
                        'y' : int(item[2]),
                        'class' : item[0]}
                imgs[-1]['pos'].append(item)
    for img in imgs:
        _fill_density(img)
    return(imgs)

def autocrop(imgs):
    """
    Crops images to the smallest rectangle including all data points
    """
    for img in imgs:
        S = img['image'].shape
        x_min = min([item['x'] for item in img['pos']])
        x_max = S[1] - 1 - max([item['x'] for item in img['pos']])
        y_min = min([item['y'] for item in img['pos']])
        y_max = S[0] - 1 - max([item['y'] for item in img['pos']])
        patch = ((y_min, y_max), (x_min, x_max))
        img['image'] = util.crop(img['image'], patch)
        img['dens'] = util.crop(img['dens'], patch)
        for item in img['pos']:
            item['x'] -= x_min
            item['y'] -= y_min

def patch_sample(N, img, height = 25, width = 25):
    """
    Randomly samples N patches from image
    """
    image = img['image']
    partial_crop = functools.partial(_patch_crop, height = height,
                                                  width = width,
                                                  img = img)
    x_pad = width // 2
    y_pad = height // 2
    sample_x = np.random.randint(x_pad, image.shape[1] - x_pad, size = N)
    sample_y = np.random.randint(y_pad, image.shape[0] - y_pad, size = N)
    sample = map(partial_crop, sample_x, sample_y)
    return(sample)

def _patch_crop(x, y, height, width, img):
    """
    Crops a given image patch centered on (x, y) while preserving data coherence
    """
    x_pad = width // 2
    y_pad = height // 2
    x_l = x - x_pad
    y_l = y - y_pad
    x_r = img['image'].shape[1] - x_l - width
    y_r = img['image'].shape[0] - y_l - height
    image = util.crop(img['image'], ((y_l, y_r), (x_l, x_r)))
    dens = util.crop(img['dens'], ((y_l, y_r), (x_l, x_r)))
    x_r = x_l + width
    y_r = y_l + width
    patch = {'path' : img['path'],
             'image' : image,
             'pos' : filter(lambda item: (x_l <= item['x'] < x_r and
                                          y_l <= item['y'] < y_r), img['pos']),
             'dens' : dens}
    patch['pos'] = map(lambda item: {'x': item['x'] - x_l,
                                     'y': item['y'] - y_l,
                                     'class': item['class']}, patch['pos'])
    return(patch)

def _fill_density(img, px=5):
    """
    Computes a density map from binary ground truth (gaussian blur)
    """
    P = np.asarray([(item['x'], item['y']) for item in img['pos']])
    data = np.ones(P[:,0].shape, np.float32)
    S = sparse.coo_matrix((data, (P[:,1], P[:,0])), img['image'].shape)
    dens = filters.gaussian(S.toarray(), px, mode='constant').astype(np.float32)
    img['dens'] = dens*150

def build_image(in_img, net, dim):
    """
    Computes whole image inference using a 25x25 patch-based network
    """
    CR = {'image':in_img['image'][:,:]}
    in_img = CR
    shape = (in_img['image'].shape[0] - dim, in_img['image'].shape[1] - dim)
    x, y = in_img['image'].shape[1], in_img['image'].shape[0]
    x_m, y_m = x - (25//2)*2, y - (25//2)*2
    B = torch.ByteTensor(in_img['image'])
    B = B.unfold(0, 25, 1)
    B = B.unfold(1, 25, 1)
    B = B.contiguous().view(-1, 25, 25)
    B = B[:,None]
    B_data = torch.utils.data.TensorDataset(B.cuda(), torch.zeros(B.size()[0], 1))
    B_loader = torch.utils.data.DataLoader(B_data, 300)
    result = np.zeros(1)
    for i, (images, truths) in enumerate(B_loader):
        images = Variable(images.float(), volatile=True)
        outputs = net(images)
        result = np.append(result, outputs.cpu().data.numpy())
        if i%100 == 99:
            print('Iter [%d/%d]' %(i+1, B.size()[0]//300))
    return(np.reshape(result[1:], (y_m, x_m)))

def test_loss(TL, criterion, net):
    """
    Computes test loss
    """
    net.eval() #Switch to inference mode in case of BatchNorm layers
    loss = []
    for i, (images, truths) in enumerate(TL):
        images = Variable(images.float(), volatile = True)
        truths = Variable(truths)
        outputs = net(images)
        loss.append(criterion(outputs, truths).data[0])
    net.train() #Switch back to training mode in case of BatchNorm layers
    return(statistics.mean(loss))
    
def tensor_loader(dataset, patches, patch_height, patch_width, net):
    """
    Data wrangling to sample patches and put them in a nice torch tensor format
    """
    data = []
    for image in dataset:
        data.append(patch_sample(patches, image, patch_height, patch_width))
    data = list(itertools.chain.from_iterable(data))
    data_i = (i['image'][np.newaxis,:,:] for i in data)
    data_truth = (util.crop(i['dens'], net.output_crop)[np.newaxis,:,:] for i in data)
    if torch.cuda.is_available():
        img_tensor = torch.from_numpy(np.stack(data_i)).cuda()
        truth_tensor = torch.from_numpy(np.stack(data_truth)).cuda()
    else:
        img_tensor = torch.from_numpy(np.stack(data_i))
        truth_tensor = torch.from_numpy(np.stack(data_truth))
    return(torch.utils.data.TensorDataset(img_tensor, truth_tensor))
