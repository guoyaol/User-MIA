from numpy.core.fromnumeric import shape
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_feature',
    dest='train_feature',
    type=str,
    required=True,
    help='path that the data',
)

parser.add_argument(
    '--victim_feature',
    dest='victim_feature',
    type=str,
    required=True,
    help='path that the data',
)

parser.add_argument(
    '--train_label',
    dest='train_label',
    type=str,
    required=True,
    help='path that the data',
)

parser.add_argument(
    '--victim_label',
    dest='victim_label',
    type=str,
    required=True,
    help='path that the data',
)

parser.add_argument("--feature_1",
                    default=1,
                    type=int,
                    help="open state for feature 1")
parser.add_argument("--feature_2",
                    default=1,
                    type=int,
                    help="open state for feature 2")
parser.add_argument("--feature_3",
                    default=1,
                    type=int,
                    help="open state for feature 3")
parser.add_argument("--feature_4",
                    default=1,
                    type=int,
                    help="open state for feature 3")
parser.add_argument("--feature_5",
                    default=1,
                    type=int,
                    help="open state for feature 3")

args = parser.parse_args()

input_dim = 0
if(args.feature_1 == 1):
    input_dim += 1
if(args.feature_2 == 1):
    input_dim += 1
if(args.feature_3 == 1):
    input_dim += 1
if(args.feature_4 == 1):
    input_dim += 1
if(args.feature_5 == 1):
    input_dim += 1

feature_list = []
if args.feature_1==1:
    feature_list.append(0)
if args.feature_2==1:
    feature_list.append(1)
if args.feature_3==1:
    feature_list.append(2)
if args.feature_4==1:
    feature_list.append(3)
if args.feature_5==1:
    feature_list.append(4)

# ---------------------------initiate GPU--------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)



train_inputs = pd.read_csv(
    args.train_feature, index_col=0)
train_labels = pd.read_csv(
    args.train_label, index_col=0)

train_inputs = train_inputs.to_numpy()
train_inputs = train_inputs[:,feature_list]
train_labels = train_labels.to_numpy().flatten()

test_inputs = pd.read_csv(
    args.victim_feature, index_col=0)
test_labels = pd.read_csv(
    args.victim_label, index_col=0)

test_inputs = test_inputs.to_numpy()
test_inputs = test_inputs[:,feature_list]
test_labels = test_labels.to_numpy().flatten()


attack_model = LGBMClassifier()

attack_model.fit(train_inputs,train_labels)


attack_outputs_train = attack_model.predict(train_inputs)


attack_outputs_test = attack_model.predict(test_inputs)



train_accuracy = accuracy_score(train_labels,attack_outputs_train)
test_accuracy = accuracy_score(test_labels,attack_outputs_test)
test_precision = precision_score(test_labels,attack_outputs_test)
test_recall = recall_score(test_labels,attack_outputs_test)

print('Train accuracy(net): %f %%' % (train_accuracy))

print('Test Accuracy(net): %f %%' % (test_accuracy))
print("precision", test_precision)
print("recall",test_recall)