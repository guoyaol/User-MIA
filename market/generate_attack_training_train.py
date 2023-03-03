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
from datasets.Market1501_lunet import Market1501_singlep_noaug
from optimizer import AdamOptimWrapper
from logger import logger
from model_lunet import LuNet
import numpy as np
import pandas as pd
from numpy.lib.function_base import average


def sum_dim(emb):
    sum_list = []
    count = emb.shape[0]

    for i in range(count):
        current_sum = np.sum(emb[i], axis=0)
        sum_list.append(current_sum)
    avg_sum = average(sum_list)
    return avg_sum


def avg_dist_pair(emb):
    count = emb.shape[0]
    dist_list = []
    for i in range(count):
        for j in range(i+1, count):
            current_dist = np.linalg.norm(emb[i]-emb[j], ord=2, keepdims=False)
            dist_list.append(current_dist)
    avg_dist = average(dist_list)
    return avg_dist

def avg_dist_center(emb):
    count = emb.shape[0]
    center = np.average(emb, axis=0)
    dist_list = []
    for i in range(count):
        current_dist = np.linalg.norm(
            emb[i]-center, ord=2, keepdims=False)
        dist_list.append(current_dist)
    avg_dist = average(dist_list)
    return avg_dist

def max_dist_center(emb):
    current_max = 0
    count = emb.shape[0]
    center = np.average(emb, axis=0)

    for i in range(count):
        current_dist = np.linalg.norm(
            emb[i]-center, ord=2, keepdims=False)
        if current_dist > current_max:
            current_max = current_dist
    return current_max

def max_dist_pair(emb):
    count = emb.shape[0]
    current_max =0
    for i in range(count):
        for j in range(i+1, count):
            current_dist = np.linalg.norm(emb[i]-emb[j], ord=2, keepdims=False)
            if current_dist> current_max:
                current_max = current_dist
    return current_max

# ----------------------load splitted cifar10 dataset-----------------------------------


model = LuNet().cuda()


df_shadow_mem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/shadow_mem.csv", index_col=0)
df_shadow_nonmem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/shadow_nonmem_new.csv", index_col=0)

index_shadow_mem = df_shadow_mem.to_numpy()
index_shadow_nonmem = df_shadow_nonmem.to_numpy()

mem_features = np.zeros((100,150,5))
nonmem_features = np.zeros((100,150,5))

with torch.no_grad():
    for count_shadow in range(100):
        print(count_shadow)

        model.load_state_dict(torch.load(
            "./saved_model_aug/"+str(count_shadow)+'.pkl'))
        print("model loaded")
        model.eval()

        current_index_mem = index_shadow_mem[count_shadow]
        current_index_nonmem = index_shadow_nonmem[count_shadow]

        for count_person in range(150):
            current_person = int(current_index_mem[count_person])
            ds_mem = Market1501_singlep_noaug(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/train_equal/'+str(current_person), is_train=False)
            imgs_currentperson = [ds_mem[i][0].tolist() for i in range(len(ds_mem))]
            imgs_currentperson = np.array(imgs_currentperson)

            current_embeddings = model(torch.tensor(imgs_currentperson).float().cuda())

            current_embeddings = current_embeddings.cpu().detach().numpy()

            current_avg_dist_center = avg_dist_center(current_embeddings)
            current_avg_dist_pair = avg_dist_pair(current_embeddings)
            current_sum_dim = sum_dim(current_embeddings)
            current_max_dist_center = max_dist_center(current_embeddings)
            current_max_dist_pair = max_dist_pair(current_embeddings)

            features=[]
            features.append(current_avg_dist_center)
            features.append(current_avg_dist_pair)
            features.append(current_sum_dim)
            features.append(current_max_dist_center)
            features.append(current_max_dist_pair)
            #上面这些代码有待验证
            mem_features[count_shadow][count_person]=np.array(features)

        for count_person in range(150):
            current_person = int(current_index_nonmem[count_person])
            ds_nonmem = Market1501_singlep_noaug(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/test_equal/'+str(current_person), is_train=False)
            imgs_currentperson = [ds_nonmem[i][0].tolist() for i in range(len(ds_nonmem))]
            imgs_currentperson = np.array(imgs_currentperson)

            current_embeddings = model(torch.tensor(imgs_currentperson).float().cuda())
            current_embeddings = current_embeddings.cpu().detach().numpy()

            current_avg_dist_center = avg_dist_center(current_embeddings)
            current_avg_dist_pair = avg_dist_pair(current_embeddings)
            current_sum_dim = sum_dim(current_embeddings)
            current_max_dist_center = max_dist_center(current_embeddings)
            current_max_dist_pair = max_dist_pair(current_embeddings)

            features=[]
            features.append(current_avg_dist_center)
            features.append(current_avg_dist_pair)
            features.append(current_sum_dim)
            features.append(current_max_dist_center)
            features.append(current_max_dist_pair)
            #上面这些代码有待验证
            nonmem_features[count_shadow][count_person]=np.array(features)

attack_input = np.vstack((mem_features,nonmem_features))
attack_input = attack_input.reshape(2*100*150,5)

attack_labels = np.hstack((np.ones(100*150),np.zeros(100*150)))


df_input = pd.DataFrame(attack_input)
df_labels = pd.DataFrame(attack_labels)
df_input.to_csv('./attack_train_set_new/attack_input_train.csv')
df_labels.to_csv('./attack_train_set_new/attack_labels_train.csv')