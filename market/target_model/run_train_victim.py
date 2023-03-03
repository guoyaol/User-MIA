#!/usr/bin/python
# -*- encoding: utf-8 -*-

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
from datasets.Market1501_lunet import Market1501
from optimizer import AdamOptimWrapper
from logger import logger
from model_lunet import LuNet


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        '--data_path',
        dest='data_path',
        type=str,
        required=True,
        help='where the training set is saved',
    )

    parse.add_argument(
        '--model_name',
        dest='model_name',
        type=str,
        required=True,
        help='where the training set is saved',
    )

    parse.add_argument(
        '--num_itr',
        dest='num_itr',
        type=int,
        default=25000,
        help='number of data iterations'
    )

    return parse.parse_args()


def train(args_data_path, args_save_path, args_num_itr, args_person_list, current_model_name):
    # setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'):
        os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    net = LuNet().cuda()
    net = nn.DataParallel(net)
    # no margin means soft-margin
    triplet_loss = TripletLoss(margin=None).cuda()

    # optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(net.parameters(), lr=3e-4,
                             wd=0, t0=15000, t1=25000)

    # dataloader
    selector = BatchHardTripletSelector()
    ds = Market1501(
        args_data_path, args_person_list, is_train=True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=4)
    diter = iter(dl)

    # train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()

        #print("----shape of input----")
        # print(imgs.shape)

        embds = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, lr: {:4f}, time: {:3f}'.format(
                count, loss_avg, optim.lr, time_interval))
            loss_avg = []
            t_start = t_end

        count += 1
        if count == args_num_itr:
            torch.save(net.module.state_dict(), os.path.join(
                args_save_path, current_model_name))
            break

    # dump model
    """ logger.info('saving trained model')
    torch.save(net.module.state_dict(), os.path.join(
        args.save_path, 'model.pkl')) """

    logger.info('everything finished')


if __name__ == '__main__':
    args = parse_args()
    import pandas as pd


    df_victim_mem = pd.read_csv(
        "/root/Shadow_Attack/datasets/Market-1501-2/shadow_attack/victim_mem.csv", index_col=0)

    victim_mem = df_victim_mem.to_numpy()
    victim_mem = victim_mem.reshape(150,)

    train(args.data_path,
          '/root/Shadow_Attack/saved_model_aug', args.num_itr, victim_mem, args.model_name)
