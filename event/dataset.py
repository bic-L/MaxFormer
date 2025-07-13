import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from torchvision import transforms
import torch
import random
from torch.utils.data.dataloader import default_collate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.asl_dvs import ASLDVS 
from spikingjelly.datasets.nav_gesture import NAVGestureWalk
from spikingjelly.datasets.nav_gesture import NAVGestureSit
from augment import EventAugment


def Cifar10DVS(root, resolution=(128, 128)):
    dataset = CIFAR10DVS(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=10)
    return SpikingjellyDataset(train_set, True, resolution=resolution), SpikingjellyDataset(test_set, False, resolution=resolution)

def Dvs128Gesture(root, resolution=(128, 128)):
    train_set = DVS128Gesture(root, train=True, data_type="event")
    test_set =  DVS128Gesture(root, train=False, data_type="event")
    return SpikingjellyDataset(train_set, True, resolution=resolution), SpikingjellyDataset(test_set, False, resolution=resolution)

def AslDVS(root, saltnoise):
    dataset = ASLDVS(root, data_type="event")
    print(len(dataset))
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=24)
    return SpikingjellyDataset(train_set, True, resolution=(180, 240), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(180, 240), saltnoise=saltnoise)

def NavGestureWalk(root, saltnoise):
    dataset = NAVGestureWalk(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.7, dataset, num_classes=6)
    return SpikingjellyDataset(train_set, True, resolution=(240, 304), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(304, 240), saltnoise=saltnoise)

def NavGestureSit(root, saltnoise):
    dataset = NAVGestureSit(root, data_type="event")
    train_set, test_set = split_to_train_test_set(0.9, dataset, num_classes=6)
    return SpikingjellyDataset(train_set, True, resolution=(240, 304), saltnoise=saltnoise), SpikingjellyDataset(test_set, False, resolution=(304, 240), saltnoise=saltnoise)

import torch.nn.functional as F


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        max_num = np.max(img)
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      
        mask = np.repeat(mask, c, axis=2)                                               
        img[mask == 0] = 0                                                              
        img[mask == 1] = max_num                                                          
        img = torch.tensor(img)
        return img
    
class FlipNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)                                                             # 图片转numpy
        l, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.zeros((l, c))
        random_mask = np.random.choice((0, 1), size=(l, 1), p=[Sd, Nd])
        mask[:, 3] = random_mask[:, 0]
        img[mask == 1] = (img[mask == 1] + 1) % 2
        img = torch.tensor(img)
        return img

class SpikingjellyDataset:

    def __init__(self, dataset, train, resolution):
        self.dataset = dataset
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None
        self.quantization_layer = QuantizationLayerVoxGrid((9,  128, 128))
        self.crop_dimension = (128, 128)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dict_events, label = self.dataset[idx]

        x = dict_events['x'].astype(np.float32)
        y = dict_events['y'].astype(np.float32)
        t = dict_events['t'].astype(np.float32)
        p = dict_events['p'].astype(np.float32)
        
        events = torch.from_numpy(np.concatenate([x[:, np.newaxis], y[:, np.newaxis], t[:, np.newaxis], p[:, np.newaxis]], axis=1))
        # print(events.shape)
        # print(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        events =torch.cat([events, torch.zeros(len(events), 1)], dim=1)
        vox = self.quantization_layer.forward(events)
        events = self.resize_to_resolution(vox)
        events = events.squeeze(0)

        return events, label

    def resize_to_resolution(self, x):
        B, C, H, W = x.shape
        if H > W:
            ZeroPad = nn.ZeroPad2d(padding=(int((H - W) / 2), int((H - W) / 2), 0, 0))
        else:
            ZeroPad = nn.ZeroPad2d(padding=(0, 0, int((W - H) / 2), int((W - H) / 2)))
        y = ZeroPad(x)
        y = F.interpolate(y, size=self.crop_dimension)
        return y

class Loader:
    def __init__(self, dataset, args, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        self.sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, sampler=self.sampler,
                                             num_workers=args.train_num_workers, pin_memory=True,
                                             collate_fn=collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = torch.cat([d[0], i*torch.ones((len(d[0]), 1), dtype=torch.float32)], 1)
        events.append(ev)
    events = torch.cat(events, 0)
    labels = default_collate(labels)
    return events, labels


class NCaltech101:
    def __init__(self, root, train, resolution):
        self.classes = listdir(root)
        self.classes.sort()
        self.files = []
        self.labels = []
        if train:
            self.event_augment = EventAugment(resolution)
        else:
            self.event_augment = None
        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)
        self.np_labels = np.array(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        f = self.files[idx]
        events = np.load(f).astype(np.float32)
        events[:, 3] = (events[:, 3] + 1) / 2
        events = torch.from_numpy(events)
        if self.event_augment is not None and random.random() < 0.5:
            events = self.event_augment(events)
        return events, label


import torch.nn as nn
    
class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        # print({'ici!'})
        # print(len(events))
        epsilon = 10e-3
        B = int(1+events[-1, -1].item())
        # tqdm.write(str(B))
        num_voxels = int(2 * np.prod(self.dim) * B)
        C, H, W = self.dim
        # print(C,H,W)
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        # get values for each channel
        x, y, t, p, b = events.T
        for bi in range(B):
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b
        for i_bin in range(C):
            values = torch.zeros_like(t)
            values[(t > i_bin/C) & (t <= (i_bin+1)/C)] = 1

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1) # (B, 2, H, W)
        return vox
    

