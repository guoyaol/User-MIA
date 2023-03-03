from cv2 import CAP_PROP_XI_COLUMN_FPN_CORRECTION
from numpy.core.fromnumeric import shape
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

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

args = parser.parse_args()


train_inputs = pd.read_csv(
    args.train_feature, index_col=0)
train_labels = pd.read_csv(
    args.train_label, index_col=0)

train_inputs = train_inputs.to_numpy()
train_labels = train_labels.to_numpy().flatten()

test_inputs = pd.read_csv(
    args.victim_feature, index_col=0)
test_labels = pd.read_csv(
    args.victim_label, index_col=0)

test_inputs = test_inputs.to_numpy()
test_labels = test_labels.to_numpy().flatten()



attack_model = LGBMClassifier()

attack_model.fit(train_inputs,train_labels)


attack_outputs_train = attack_model.predict(train_inputs)


attack_outputs_test = attack_model.predict(test_inputs)



train_accuracy = accuracy_score(train_labels,attack_outputs_train)
test_accuracy = accuracy_score(test_labels,attack_outputs_test)

test_precision = precision_score(test_labels,attack_outputs_test)
test_recall = recall_score(test_labels,attack_outputs_test)

print(attack_outputs_test.shape)
shaped_result = attack_outputs_test.reshape((300,8))
vote_labels = np.hstack((np.ones(150),np.zeros(150)))
print(shaped_result)

voted_results = []

#voting

for count_person in range(shaped_result.shape[0]):
    current_person_mem =0
    current_person_nonmem=0
    current_person_results = shaped_result[count_person]
    for count_image in range(shaped_result.shape[1]):
        if shaped_result[count_person][count_image]==1:
            current_person_mem +=1
        elif shaped_result[count_person][count_image]==0:
            current_person_nonmem +=1
    assert current_person_nonmem+current_person_mem == 8
    if current_person_mem > 8/2:
        voted_results.append(1)
    else:
        voted_results.append(0)

vote_accuracy = accuracy_score(vote_labels,np.array(voted_results))

vote_precision = precision_score(vote_labels,np.array(voted_results))
vote_recall = recall_score(vote_labels,np.array(voted_results))



print('Train accuracy(net): %f %%' % (train_accuracy))

print('Test Accuracy(net): %f %%' % (test_accuracy))

print('test precision', test_precision)
print('test recall', test_recall)

print('vote accuracy',vote_accuracy)
print('vote precision', vote_precision)
print('voten recall', vote_recall)