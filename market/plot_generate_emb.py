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
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

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
import torchvision.transforms as transforms

# ----------------------load splitted cifar10 dataset-----------------------------------

class Market1501_plot(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(Market1501_plot, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        print(self.imgs)
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        if is_train:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
            ])
        else:

            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
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


model = LuNet().cuda()


df_victim_mem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/victim_mem.csv", index_col=0)
df_victim_nonmem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/victim_nonmem_new.csv", index_col=0)

index_victim_mem = df_victim_mem.to_numpy()
index_victim_nonmem = df_victim_nonmem.to_numpy()
index_victim_mem = index_victim_mem.reshape(150,)
index_victim_nonmem = index_victim_nonmem.reshape(150,)

mem_features = np.zeros((150,5))
nonmem_features = np.zeros((150,5))

with torch.no_grad():

    model.load_state_dict(torch.load(
        "./saved_model_aug/"+'victim.pkl'))
    print("model loaded")
    model.eval()


    current_person = 152
    ds_mem = Market1501_plot(
        '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/test_split/'+str(current_person), is_train=False)
    imgs_currentperson = [ds_mem[i][0].tolist() for i in range(len(ds_mem))]
    imgs_currentperson = np.array(imgs_currentperson)

    current_embeddings = model(torch.tensor(imgs_currentperson).float().cuda())

    current_embeddings = current_embeddings.cpu().detach().numpy()

    print(current_embeddings)
    df_input = pd.DataFrame(current_embeddings)
    df_input.to_csv('./plot/152_test.csv')


""" attack_input = np.vstack((mem_features,nonmem_features))
attack_input = attack_input.reshape(2*150,5)

attack_labels = np.hstack((np.ones(150),np.zeros(150)))


df_input = pd.DataFrame(attack_input)
df_labels = pd.DataFrame(attack_labels)
df_input.to_csv('./attack_train_set_new/victim_input.csv')
df_labels.to_csv('./attack_train_set_new/victim_labels.csv') """