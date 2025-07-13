import copy
import os.path

import numpy as np
import math
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import cv2


class EventAugment(object):
    def __init__(self, resolution):
        self.resolution = resolution
        self.augment_list = [
            (self.identity, 0, 0),
            (self.drop_by_time, 0.1, 0.9),
            (self.drop_by_area, 0.1, 0.5),
            (self.random_drop, 0.1, 0.5),
            # (self.drop_by_area_with_cam, 0.1, 0.6),
            # (self.random_drop_with_cam, 0.5, 1),
            (self.overall_noise, 0.1, 0.9),
            (self.region_noise, 0.1, 0.5),
            # (self.overall_noise_with_cam, 0.1, 1),
            # (self.region_noise_with_cam, 0.1, 0.9),
            (self.time_incline_x, 0.05, 0.5),
            (self.time_incline_y, 0.05, 0.5),
           #  (self.random_shift_time, 0.1, 0.8),

            (self.random_shift_xy, 1, 10),
            (self.flip_along_x, 0, 0),
            (self.flip_along_y, 0, 0),
            (self.flip_along_time, 0, 0),
            (self.rotate, 0, math.pi / 2),
            (self.linear_x, 0, 0.6),
            (self.linear_y, 0, 0.6),
            (self.shear_x, 0, 1),
            (self.shear_y, 0, 1),
            (self.scale, 0.2, 2)]
        self.ops_name = []
        self.ops_list = []
        self.mags_list = []
        self.l_ops = len(self.augment_list)
        self.l_uniq = 0
        for idx, op in enumerate(self.augment_list):
            self.ops_name.append(op.__str__().split(' ')[2].split('.')[1])

    def __call__(self, events):
        op_idx = random.randint(0, len(self.augment_list)) - 1 
        op_max = self.augment_list[op_idx][2]
        op_min = self.augment_list[op_idx][1]
        op = self.augment_list[op_idx][0]
        aug_events = op(events, random.random() * (op_max - op_min) + op_min)
        return aug_events

    def identity(self, events, v):
        events = copy.deepcopy(events)
        return events
    


    def overall_noise(self, events, ratio):
        events = copy.deepcopy(events).to(events.device)
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        len_noise = int(len(events) * ratio)
        x_noise = torch.randint(high=self.resolution[1], size=(len_noise, 1))
        y_noise = torch.randint(high=self.resolution[0], size=(len_noise, 1))
        t_noise = torch.rand(size=(len_noise, 1)) * (t_max - t_min) + t_min
        p_noise = torch.randint(high=2, size=(len_noise, 1))
        noise_events = torch.cat([x_noise, y_noise, t_noise, p_noise], dim=1).to(events.device)
        return torch.cat([events, noise_events])

    def region_noise(self, events, area_ratio):
        events = copy.deepcopy(events).to(events.device)
        length_scale = torch.rand(1) + 0.5
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        x0 = np.random.uniform(self.resolution[1])
        y0 = np.random.uniform(self.resolution[0])
        x_out = self.resolution[1] * area_ratio * length_scale
        y_out = self.resolution[0] * area_ratio / length_scale
        x0 = int(max(0, x0 - x_out / 2.0))
        y0 = int(max(0, y0 - y_out / 2.0))
        x1 = int(min(self.resolution[1], x0 + x_out))
        y1 = int(min(self.resolution[0], y0 + y_out))
        len_noise = int(len(events) * area_ratio ** 2)
        x_noise = torch.randint(low=x0, high=x1, size=(len_noise, 1))
        y_noise = torch.randint(low=y0, high=y1, size=(len_noise, 1))
        t_noise = torch.rand(len_noise, 1) * (t_max - t_min) + t_min
        p_noise = torch.randint(high=2, size=(len_noise, 1))
        noise_events = torch.cat([x_noise, y_noise, t_noise, p_noise], dim=1).to(events.device)
        return torch.cat([events, noise_events])



    def random_shift_time(self, events, max_shift_ratio):
        events = copy.deepcopy(events)
        max_shift_ratio = int(max_shift_ratio)
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        shift_length = max_shift_ratio * (t_max - t_min)
        t_shift = (torch.rand(size=(len(events),)).to(events.device) - 0.5) * shift_length
        events[:, 2] += t_shift
        return events

    def random_shift_xy(self, events, max_shift_length):
        events = copy.deepcopy(events)
        H, W = self.resolution
        max_shift_length = int(max_shift_length)
        x_shift, y_shift = torch.randint(low=-max_shift_length, high=max_shift_length + 1, size=(2, len(events))).to(
            events.device)
        events[:, 0] += x_shift
        events[:, 1] += y_shift
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def flip_along_x(self, events, v):
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 0] = W - 1 - events[:, 0]
        return events

    def flip_along_y(self, events, v):
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 1] = H - 1 - events[:, 1]
        return events

    def rotate(self, events, theta):
        events = copy.deepcopy(events)
        H, W = self.resolution
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        x = events[:, 0] - x_mid
        y = events[:, 1] - y_mid
        events[:, 0] = torch.round(x * math.cos(theta) + y * math.sin(theta) + x_mid)
        events[:, 1] = torch.round(-x * math.sin(theta) + y * math.cos(theta) + y_mid)
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def linear_x(self, events, linear):
        events = copy.deepcopy(events)
        W = self.resolution[1]
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        x_mid = (x_max + x_min) / 2
        if linear > 0:
            linear_w = int(linear * (W - x_mid))
        else:
            linear_w = int(linear * x_mid)
        events[:, 0] = events[:, 0] + linear_w
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W)
        return events[valid_events]

    def linear_y(self, events, linear):
        events = copy.deepcopy(events)
        H = self.resolution[0]
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        y_mid = (y_max + y_min) / 2
        if linear > 0:
            linear_h = int(linear * (H - y_mid))
        else:
            linear_h = int(linear * y_mid)
        events[:, 1] = events[:, 1] + linear_h
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def drop_by_time(self, events, ratio):
        events = copy.deepcopy(events)
        timestamps = events[:, 2]
        t_max = timestamps.max()
        t_min = timestamps.min()
        t_period = t_max - t_min
        drop_period = t_period * ratio
        t_start = torch.rand(1).to(events.device) * (t_max - drop_period - t_min) + t_min
        t_end = t_start + drop_period
        idx = (timestamps < t_start) | (timestamps > t_end)
        if events[idx].shape[0] == 0:
            return events
        return events[idx]

    def drop_by_area(self, events, area_ratio):
        events = copy.deepcopy(events)
        length_scale = torch.rand(1).to(events.device) + 0.5
        x0 = np.random.uniform(self.resolution[1])
        y0 = np.random.uniform(self.resolution[0])
        x_out = self.resolution[1] * area_ratio * length_scale
        y_out = self.resolution[0] * area_ratio / length_scale
        x0 = int(max(0, x0 - x_out / 2.0))
        y0 = int(max(0, y0 - y_out / 2.0))
        x1 = min(self.resolution[1], x0 + x_out)
        y1 = min(self.resolution[0], y0 + y_out)
        xy = (x0, x1, y0, y1)
        idx1 = (events[:, 0] < xy[0]) | (events[:, 0] > xy[1])
        idx2 = (events[:, 1] < xy[2]) | (events[:, 1] > xy[3])
        idx = idx1 | idx2
        if events[idx].shape[0] != 0:
            return events[idx]
        else:
            return events

    def random_drop(self, events, ratio):
        events = copy.deepcopy(events)
        N = events.shape[0]
        num_drop = int(N * ratio)
        idx = random.sample(list(np.arange(0, N)), N - num_drop)
        return events[idx]

    def drop_by_area_with_cam(self, events, area_ratio):
        events = copy.deepcopy(events)
        cam_areas = self.rel_cam.get_threshold(events)
        B = int(events[-1, -1].item() + 1)
        aug_events = []
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            x_min, y_min, x_max, y_max = cam_areas[b]
            x0 = np.random.uniform(x_min, x_max)
            y0 = np.random.uniform(y_min, y_max)
            x_out = (x_max - x_min) * area_ratio
            y_out = (y_max - y_min) * area_ratio
            x0 = int(max(0, x0 - x_out / 2.0))
            y0 = int(max(0, y0 - y_out / 2.0))
            x1 = min(x_max, x0 + x_out)
            y1 = min(y_max, y0 + y_out)
            xy = (x0, x1, y0, y1)
            idx1 = (single_events[:, 0] < xy[0]) | (single_events[:, 0] > xy[1])
            idx2 = (single_events[:, 1] < xy[2]) | (single_events[:, 1] > xy[3])
            idx = idx1 | idx2
            if single_events[idx].shape[0] != 0:
                single_events = single_events[idx]
            single_events = torch.cat(
                [single_events, b * torch.ones((len(single_events), 1), dtype=torch.float32).to(single_events.device)], 1)
            aug_events.append(single_events)
        aug_events = torch.cat(aug_events, 0)
        return aug_events

    def random_drop_with_cam(self, events, lamda):
        events = copy.deepcopy(events)
        cam_probs = self.rel_cam.get_heat_prob(events, str_target_layer="long")
        cam_probs = cam_probs * lamda
        B = int(events[-1, -1].item() + 1)
        aug_events = []
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            cam_prob = cam_probs[b]
            rand = torch.randn(len(single_events)).to(single_events.device)
            int_events = single_events[:, :2].to(torch.int64)
            t_max, t_min = single_events[:, 2].max(), single_events[:, 2].min()
            num_channel = int(cam_prob.shape[0] / 2)
            len_channel = ((t_max - t_min) / num_channel).item()
            t = ((single_events[:, 2] - t_min).div(len_channel, rounding_mode='floor')).clamp(max=num_channel - 1).to(torch.int64)
            p = single_events[:, 3].to(torch.int64)
            index = rand[:] > cam_prob[t + p * num_channel, int_events[:, 1], int_events[:, 0]]
            if single_events[index].shape[0] != 0:
                single_events = single_events[index]
            single_events = torch.cat(
                [single_events, b * torch.ones((len(single_events), 1), dtype=torch.float32).to(single_events.device)],
                1)
            aug_events.append(single_events)
        aug_events = torch.cat(aug_events, 0)
        return aug_events

    def overall_noise(self, events, ratio):
        events = copy.deepcopy(events).to(events.device)
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        len_noise = int(len(events) * ratio)
        x_noise = torch.randint(high=self.resolution[1], size=(len_noise, 1))
        y_noise = torch.randint(high=self.resolution[0], size=(len_noise, 1))
        t_noise = torch.rand(size=(len_noise, 1)) * (t_max - t_min) + t_min
        p_noise = torch.randint(high=2, size=(len_noise, 1))
        noise_events = torch.cat([x_noise, y_noise, t_noise, p_noise], axis=1).to(events.device)
        return torch.cat([events, noise_events])

    def region_noise_with_cam(self, events, area_ratio):
        events = copy.deepcopy(events).to(events.device)
        cam_areas = self.rel_cam.get_threshold(events)
        B = int(events[-1, -1].item() + 1)
        aug_events = []
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            length_scale = torch.rand(1) + 0.5
            x_min, y_min, x_max, y_max = cam_areas[b]
            t_max = torch.amax(single_events[:, 2]).item()
            t_min = torch.amin(single_events[:, 2]).item()
            W = x_max - x_min
            H = y_max - y_min
            x0 = np.random.uniform(low=x_min, high=x_max)
            y0 = np.random.uniform(low=y_min, high=y_max)
            x_out = W * area_ratio * length_scale
            y_out = H * area_ratio / length_scale
            x0 = int(max(0, x0 - x_out / 2.0))
            y0 = int(max(0, y0 - y_out / 2.0))
            x1 = int(min(self.resolution[1], x0 + x_out))
            y1 = int(min(self.resolution[0], y0 + y_out))
            len_noise = int(len(single_events) * area_ratio ** 2)
            x_noise = torch.randint(low=x0, high=x1, size=(len_noise, 1))
            y_noise = torch.randint(low=y0, high=y1, size=(len_noise, 1))
            t_noise = torch.rand(len_noise, 1) * (t_max - t_min) + t_min
            p_noise = torch.randint(high=2, size=(len_noise, 1))
            noise_events = torch.cat([x_noise, y_noise, t_noise, p_noise], dim=1).to(single_events.device)
            single_events = torch.cat([single_events, noise_events])
            single_events = torch.cat(
                [single_events, b * torch.ones((len(single_events), 1), dtype=torch.float32).to(single_events.device)],
                1)
            aug_events.append(single_events)
        aug_events = torch.cat(aug_events, 0)
        return aug_events

    def overall_noise_with_cam(self, events, noise_ratio):
        events = copy.deepcopy(events)
        H, W = self.resolution
        cam_probs = self.rel_cam.get_heat_prob(events, str_target_layer='layer4')
        B = int(events[-1, -1].item() + 1)
        aug_events = []
        for b in range(B):
            single_events = events[events[:, 4] == b, :4]
            cam_prob = cam_probs[b]
            t_max = torch.amax(single_events[:, 2]).item()
            t_min = torch.amin(single_events[:, 2]).item()
            len_noise = int(noise_ratio * len(single_events))

            cam_prob = cam_prob.view(-1)
            cam_prob = cam_prob / (torch.sum(cam_prob).item() + 1e-9)
            if torch.isnan(cam_prob).sum() != 0:
                print("NaN Checked")
            else:
                max_index = torch.argmax(cam_prob)
                cam_prob = cam_prob.tolist()
                cam_sum = sum(cam_prob)
                cam_prob[max_index] += 1 - cam_sum
                index = [i for i in range(H * W)]
                xy = np.random.choice(index, [len_noise], p=cam_prob)
                x_noise = xy % W
                y_noise = xy // W
                t_noise = np.random.random(len_noise) * (t_max - t_min) + t_min
                p_noise = np.random.randint(low=0, high=2, size=len_noise)
                noise_events = torch.tensor(np.array([x_noise, y_noise, t_noise, p_noise], dtype=np.float32).T).to(single_events.device)
                single_events = torch.cat([single_events, noise_events])
            single_events = torch.cat(
                [single_events, b * torch.ones((len(single_events), 1), dtype=torch.float32).to(single_events.device)],
                1)
            aug_events.append(single_events)
        aug_events = torch.cat(aug_events, 0)
        return aug_events


    def time_incline_x(self, events, kx):
        events = copy.deepcopy(events)
        H, W = self.resolution
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        events[:, 2] = events[:, 2] + (events[:, 0] - W / 2) * kx / W * (t_max - t_min)
        events[:, 2] = events[:, 2] - events[:, 2].min()
        return events

    def time_incline_y(self, events, ky):
        events = copy.deepcopy(events)
        H, W = self.resolution
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        events[:, 2] = events[:, 2] + (events[:, 1] - H / 2) * ky / H * (t_max - t_min)
        events[:, 2] = events[:, 2] - events[:, 2].min()
        return events

    def random_shift_time(self, events, max_shift_ratio):
        events = copy.deepcopy(events)
        max_shift_ratio = int(max_shift_ratio)
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        shift_length = max_shift_ratio * (t_max - t_min)
        t_shift = (torch.rand(size=(len(events),)).to(events.device) - 0.5) * shift_length
        events[:, 2] += t_shift
        return events

    def random_shift_xy(self, events, max_shift_length):
        events = copy.deepcopy(events)
        H, W = self.resolution
        max_shift_length = int(max_shift_length)
        x_shift, y_shift = torch.randint(low=-max_shift_length, high=max_shift_length + 1, size=(2, len(events))).to(
            events.device)
        events[:, 0] += x_shift
        events[:, 1] += y_shift
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def flip_along_x(self, events, v):
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 0] = W - 1 - events[:, 0]
        return events

    def flip_along_y(self, events, v):
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 1] = H - 1 - events[:, 1]
        return events

    def flip_along_time(self, events, v):
        events = copy.deepcopy(events)
        t_max = torch.amax(events[:, 2]).item()
        t_min = torch.amin(events[:, 2]).item()
        events[:, 2] = (t_max - events[:, 2]) + t_min
        return events

    def rotate(self, events, theta):
        if random.random() < 0.5:
            theta = -theta
        events = copy.deepcopy(events)
        H, W = self.resolution
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        x = events[:, 0] - x_mid
        y = events[:, 1] - y_mid
        events[:, 0] = torch.round(x * math.cos(theta) + y * math.sin(theta) + x_mid)
        events[:, 1] = torch.round(-x * math.sin(theta) + y * math.cos(theta) + y_mid)
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def linear_x(self, events, linear):
        if random.random() < 0.5:
            linear = -linear
        events = copy.deepcopy(events)
        W = self.resolution[1]
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        x_mid = (x_max + x_min) / 2
        if linear > 0:
            linear_w = int(linear * (W - x_mid))
        else:
            linear_w = int(linear * x_mid)
        events[:, 0] = events[:, 0] + linear_w
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W)
        return events[valid_events]

    def linear_y(self, events, linear):
        if random.random() < 0.5:
            linear = -linear
        events = copy.deepcopy(events)
        H = self.resolution[0]
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        y_mid = (y_max + y_min) / 2
        if linear > 0:
            linear_h = int(linear * (H - y_mid))
        else:
            linear_h = int(linear * y_mid)
        events[:, 1] = events[:, 1] + linear_h
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def shear_x(self, events, shear):
        if random.random() < 0.5:
            shear = -shear
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        y_mid = (y_max + y_min) / 2
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 0] = torch.round(events[:, 0] + shear * (events[:, 1] - y_mid) / (y_max - y_min) * (x_max - x_min))
        valid_events = (events[:, 0] >= 0) & (events[:, 0] < W)
        return events[valid_events]

    def shear_y(self, events, shear):
        if random.random() < 0.5:
            shear = -shear
        x_min, x_max = events[:, 0].min().item(), events[:, 0].max().item()
        y_min, y_max = events[:, 1].min().item(), events[:, 1].max().item()
        x_mid = (x_max + x_min) / 2
        events = copy.deepcopy(events)
        H, W = self.resolution
        events[:, 1] = torch.round(events[:, 1] + shear * (events[:, 0] - x_mid) / (x_max - x_min) * (y_max - y_min))
        valid_events = (events[:, 1] >= 0) & (events[:, 1] < H)
        return events[valid_events]

    def scale(self, events, factor):
        scale_events = copy.deepcopy(events)
        H, W = self.resolution
        x_min, x_max = scale_events[:, 0].min().item(), scale_events[:, 0].max().item()
        y_min, y_max = scale_events[:, 1].min().item(), scale_events[:, 1].max().item()
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        scale_events[:, 0] = torch.round((scale_events[:, 0] - x_mid) * factor + x_mid)
        scale_events[:, 1] = torch.round((scale_events[:, 1] - y_mid) * factor + y_mid)
        valid_events = (scale_events[:, 0] >= 0) & (scale_events[:, 0] < W) & (scale_events[:, 1] >= 0) & (scale_events[:, 1] < H)
        scale_events = scale_events[valid_events]
        if scale_events.shape[0] == 0:
            return events
        return scale_events

    def event_drop(self, events):
        raw_events = events
        option = random.randint(0, 4)  # 0: identity, 1: drop_by_time, 2: drop_by_area, 3: random_drop
        if option == 0:  # identity, do nothing
            return events
        elif option == 1:  # drop_by_time
            T = random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
            events = self.drop_by_time(events, ratio=T)
        elif option == 2:  # drop by area
            area_ratio = random.randint(1, 6) / 20.0  # np.random.uniform(0.05, 0.1, 0.15, 0.2, 0.25)
            events = self.drop_by_area(events, area_ratio=area_ratio)
        elif option == 3:  # random drop
            ratio = random.randint(1, 6) / 10.0  # np.random.uniform(0.1, 0.9)
            events = self.random_drop(events, ratio=ratio)
        if len(events) == 0:  # avoid dropping all the events
            events = raw_events
        return events
