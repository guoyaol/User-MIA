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


class prid2011_base_strong(Dataset):
    '''
    a wrapper of PRID_2011 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(prid2011_base_strong, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = []
        self.lb_ids=[]


        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.png']
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        if data_path[-1]=='/':
            output = data_path.split('/')[-2]
        else:
            output = data_path.split('/')[-1]
        output = int(output.split('_')[-1])
        p_id = output

        self.lb_ids = [p_id for el in self.imgs]


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
        return img,self.lb_ids[idx]

class prid2011_base_middle(Dataset):
    '''
    a wrapper of PRID_2011 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(prid2011_base_middle, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = []
        self.lb_ids=[]


        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.png']
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        if data_path[-1]=='/':
            output = data_path.split('/')[-2]
        else:
            output = data_path.split('/')[-1]
        output = int(output.split('_')[-1])
        p_id = output

        self.lb_ids = [p_id for el in self.imgs]

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
        return img,self.lb_ids[idx]

class prid2011_base_weak(Dataset):
    '''
    a wrapper of PRID_2011 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(prid2011_base_weak, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.data_path = data_path
        self.imgs = []
        self.lb_ids=[]


        self.imgs = os.listdir(data_path)
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.png']
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]

        if data_path[-1]=='/':
            output = data_path.split('/')[-2]
        else:
            output = data_path.split('/')[-1]
        output = int(output.split('_')[-1])
        p_id = output

        self.lb_ids = [p_id for el in self.imgs]

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
        return img,self.lb_ids[idx]


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
    '--training_images',
    dest='training_images',
    type=int,
    required=True,
    help='whether use training images to attack',
)

args = parser.parse_args()


model = LuNet().cuda()


df_victim_mem = pd.read_csv(
    "/root/Attack_prid2011/datasets/prid_2011_plan2/shadow_split_record/victim_mem.csv", index_col=0)
df_victim_nonmem = pd.read_csv(
    "/root/Attack_prid2011/datasets/prid_2011_plan2/shadow_split_record/victim_nonmem_new.csv", index_col=0)

index_victim_mem = df_victim_mem.to_numpy()
index_victim_nonmem = df_victim_nonmem.to_numpy()
index_victim_mem = index_victim_mem.reshape(150,)
index_victim_nonmem = index_victim_nonmem.reshape(150,)

mem_vector = np.zeros((150,10,45))
nonmem_vector = np.zeros((150,10,45))

with torch.no_grad():

    model.load_state_dict(torch.load(
        "./saved_models_aug/"+'victim.pkl'))
    model.eval()
    print("model loaded")


    for count_person in range(150):
        print(count_person)
        current_person = int(index_victim_mem[count_person])
        str_current_person = "person_"+str(current_person).zfill(4)
        if args.training_images == 1 and args.strong_attack == 1:
            ds_mem = prid2011_base_strong(
                './datasets/prid_2011_plan2/train_split_equal/'+str_current_person)
        elif args.training_images == 1 and args.strong_attack == 0:
            ds_mem = prid2011_base_weak(
                './datasets/prid_2011_plan2/train_split_equal/'+str_current_person) 
        elif args.training_images == 0 and args.strong_attack == 1:
            ds_mem = prid2011_base_strong(
                './datasets/prid_2011_plan2/semi_split/'+str_current_person)      
        elif args.training_images == 0 and args.strong_attack == 0:
            ds_mem = prid2011_base_weak(
                './datasets/prid_2011_plan2/semi_split/'+str_current_person) 
        elif args.training_images == 0 and args.strong_attack == 2:
            ds_mem = prid2011_base_middle(
                './datasets/prid_2011_plan2/semi_split/'+str_current_person)  
        elif args.training_images == 1 and args.strong_attack == 2:
            ds_mem = prid2011_base_middle(
                './datasets/prid_2011_plan2/train_split_equal/'+str_current_person) 


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
        current_person = int(index_victim_nonmem[count_person])
        str_current_person = "person_"+str(current_person).zfill(4)
        if args.strong_attack == 1:
            ds_nonmem = prid2011_base_strong(
                './datasets/prid_2011_plan2/test_split_equal/'+str_current_person, is_train=False)
        elif args.strong_attack == 0:
            ds_nonmem = prid2011_base_weak(
                './datasets/prid_2011_plan2/test_split_equal/'+str_current_person, is_train=False)
        elif args.strong_attack == 2:
            ds_nonmem = prid2011_base_middle(
                './datasets/prid_2011_plan2/test_split_equal/'+str_current_person, is_train=False)

        for count_img in range(len(ds_nonmem)):
            aug_current_imgs = np.zeros((10,3,128,64))
            for j in range(10):
                aug_current_imgs[j] = np.array(ds_nonmem[count_img][0].tolist())
            current_embeddings = model(torch.tensor(aug_current_imgs).float().cuda())
            current_embeddings = current_embeddings.cpu().detach().numpy()

            distance_vector = cal_dist_vector(current_embeddings)

            nonmem_vector[count_person][count_img] = np.array(distance_vector)


attack_input = np.vstack((np.array(mem_vector),np.array(nonmem_vector)))
attack_input = attack_input.reshape((2*150*10,45))

attack_labels = np.hstack((np.ones(150*10),np.zeros(150*10)))

print(attack_input.shape)
print(attack_labels.shape)



df_input = pd.DataFrame(attack_input)
df_labels = pd.DataFrame(attack_labels)
df_input.to_csv(os.path.join(args.save_path,'victim_input.csv'))
df_labels.to_csv(os.path.join(args.save_path,'victim_labels.csv')) 