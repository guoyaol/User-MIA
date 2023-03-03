from turtle import distance
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools
import argparse

from backbone import EmbedNetwork
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from optimizer import AdamOptimWrapper
from logger import logger
from model_lunet import LuNet
import numpy as np
import pandas as pd
from numpy.lib.function_base import average
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from random_erasing import RandomErasing

class Market1501_strong(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(Market1501_strong, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
            RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
        ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]

class Market1501_middle(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(Market1501_middle, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
            RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
        ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]

class Market1501_weak(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(Market1501_weak, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
            transforms.RandomGrayscale(),
            transforms.RandomResizedCrop(size=(128,64)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),

        ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]

def cal_dist_vector(emb):
    count = emb.shape[0]
    dist_list = []
    for i in range(count):
        for j in range(i+1, count):
            current_dist = np.linalg.norm(emb[i]-emb[j], ord=2, keepdims=False)
            dist_list.append(current_dist)
    dist_list = sorted(dist_list,reverse=True)
    
    return dist_list


parser = argparse.ArgumentParser()

parser.add_argument(
    '--save_path',
    dest='save_path',
    type=str,
    required=True,
    help='path that the data is saved',
)

parser.add_argument(
    '--strong_attack',
    dest='strong_attack',
    type=int,
    required=True,
    help='whether is strong attack',
)

parser.add_argument(
    '--shadow_id',
    dest='shadow_id',
    type=int,
    required=True,
    help='the id of the chosen shadow model',
)

args = parser.parse_args()



model = LuNet().cuda()


df_shadow_mem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/shadow_mem.csv", index_col=0)
df_shadow_nonmem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/shadow_nonmem_new.csv", index_col=0)

index_shadow_mem = df_shadow_mem.to_numpy()
index_shadow_nonmem = df_shadow_nonmem.to_numpy()


mem_vector = np.zeros((150,8,45))
nonmem_vector = np.zeros((150,8,45))

with torch.no_grad():
    count_shadow = args.shadow_id

    print(count_shadow)

    model.load_state_dict(torch.load(
        "./saved_model_aug/"+str(count_shadow)+'.pkl'))
    print("model loaded")
    model.eval()

    current_index_mem = index_shadow_mem[count_shadow]
    current_index_nonmem = index_shadow_nonmem[count_shadow]

    for count_person in range(150):
        print(count_person)
        current_person = int(current_index_mem[count_person])
        
        if args.strong_attack == 1:
            ds_mem = Market1501_strong(
            '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/train_equal/'+str(current_person))
        elif args.strong_attack == 0:
            ds_mem = Market1501_weak(
            '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/train_equal/'+str(current_person))
        elif args.strong_attack == 2:
            ds_mem = Market1501_middle(
            '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/train_equal/'+str(current_person))


        for count_img in range(len(ds_mem)):
            aug_current_imgs = np.zeros((10,3,128,64))
            for j in range(10):
                aug_current_imgs[j] = np.array(ds_mem[count_img][0].tolist())
            current_embeddings = model(torch.tensor(aug_current_imgs).float().cuda())
            current_embeddings = current_embeddings.cpu().detach().numpy()

            distance_vector = cal_dist_vector(current_embeddings)

            mem_vector[count_person][count_img] = np.array(distance_vector)

    for count_person in range(150):
        print(count_person)
        current_person = int(current_index_nonmem[count_person])
        if args.strong_attack == 1:
            ds_nonmem = Market1501_strong(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/test_equal/'+str(current_person), is_train=False)
        elif args.strong_attack == 0:
            ds_nonmem = Market1501_weak(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/test_equal/'+str(current_person), is_train=False)
        elif args.strong_attack == 2:
            ds_nonmem = Market1501_middle(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/test_equal/'+str(current_person), is_train=False)

        for count_img in range(len(ds_nonmem)):
            aug_current_imgs = np.zeros((10,3,128,64))
            for j in range(10):
                aug_current_imgs[j] = np.array(ds_nonmem[count_img][0].tolist())
            current_embeddings = model(torch.tensor(aug_current_imgs).float().cuda())
            current_embeddings = current_embeddings.cpu().detach().numpy()

            distance_vector = cal_dist_vector(current_embeddings)

            nonmem_vector[count_person][count_img] = np.array(distance_vector)

attack_input = np.vstack((mem_vector,nonmem_vector))
attack_input = attack_input.reshape((2*150*8,45))

attack_labels = np.hstack((np.ones(150*8),np.zeros(150*8)))

print(attack_input.shape)
print(attack_labels.shape)


df_input = pd.DataFrame(attack_input)
df_labels = pd.DataFrame(attack_labels)
df_input.to_csv(os.path.join(args.save_path,'attack_input.csv'))
df_labels.to_csv(os.path.join(args.save_path,'attack_labels.csv'))