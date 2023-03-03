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
from datasets.Market1501_lunet import Market1501_singlep
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

def embd_list(model, id_list):

    count_person = len(id_list)

    mem_features = np.zeros((count_person, 5))

    with torch.no_grad():
        current_count = 0
        for current_person in id_list:
            ds_mem = Market1501_singlep(
                '/root/Shadow_Attack/datasets/Market-1501-2/attack_image/semi_split/'+current_person, is_train=False)
            imgs_currentperson = [ds_mem[i][0].tolist()
                                for i in range(len(ds_mem))]
            imgs_currentperson = np.array(imgs_currentperson)

            current_embeddings = model(torch.tensor(
                imgs_currentperson).float().cuda())

            current_embeddings = current_embeddings.cpu().detach().numpy()

            current_avg_dist_center = avg_dist_center(current_embeddings)
            current_avg_dist_pair = avg_dist_pair(current_embeddings)
            current_sum_dim = sum_dim(current_embeddings)
            current_max_dist_center = max_dist_center(current_embeddings)
            current_max_dist_pair = max_dist_pair(current_embeddings)

            features = []
            features.append(current_avg_dist_center)
            features.append(current_avg_dist_pair)
            features.append(current_sum_dim)
            features.append(current_max_dist_center)
            features.append(current_max_dist_pair)
            # 上面这些代码有待验证
            mem_features[current_count] = np.array(features)
            current_count +=1
    return mem_features


name_list = os.listdir('./datasets/Market-1501-2/attack_image/train')

count_list = [0 for i in range(1502)]




df_victim_mem = pd.read_csv(
    "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/victim_mem.csv", index_col=0)

index_victim_mem = df_victim_mem.to_numpy()
index_victim_mem = index_victim_mem.flatten()


list_victim_mem = index_victim_mem.tolist()

pair_list = []

for each_mem in list_victim_mem:
    name_list = os.listdir("./datasets/Market-1501-2/attack_image/train_split/"+str(each_mem))   
    current_count = len(name_list)
    temp_tuple = (str(each_mem), current_count)
    pair_list.append(temp_tuple)


tier_1 = []
tier_2 = []
tier_3 = []
tier_4 = []
tier_5 = []

for each_pair in pair_list:
    if each_pair[1]>=22:
        tier_1.append(each_pair)
    elif each_pair[1]>=17:
        tier_2.append(each_pair)
    elif each_pair[1]>=14:
        tier_3.append(each_pair)
    elif each_pair[1]>=11:
        tier_4.append(each_pair)
    elif each_pair[1]>=8:
        tier_5.append(each_pair)


id_tier_1 = [i[0] for i in tier_1]
id_tier_2 = [i[0] for i in tier_2]
id_tier_3 = [i[0] for i in tier_3]
id_tier_4 = [i[0] for i in tier_4]
id_tier_5 = [i[0] for i in tier_5]


model = LuNet().cuda()

with torch.no_grad():

    model.load_state_dict(torch.load("./saved_model_aug/"+'victim.pkl'))
    model.eval()
    output_tier_5 = embd_list(model, id_tier_5)
    output_tier_4 = embd_list(model, id_tier_4)
    output_tier_3 = embd_list(model, id_tier_3)
    output_tier_2 = embd_list(model, id_tier_2)
    output_tier_1 = embd_list(model, id_tier_1)


df_input = pd.DataFrame(output_tier_5)
df_input.to_csv('./tiers/tier_5.csv')

df_input = pd.DataFrame(output_tier_4)
df_input.to_csv('./tiers/tier_4.csv')

df_input = pd.DataFrame(output_tier_3)
df_input.to_csv('./tiers/tier_3.csv')

df_input = pd.DataFrame(output_tier_2)
df_input.to_csv('./tiers/tier_2.csv')

df_input = pd.DataFrame(output_tier_1)
df_input.to_csv('./tiers/tier_1.csv')
