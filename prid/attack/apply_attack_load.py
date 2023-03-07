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
from sklearn.metrics import accuracy_score, precision_score, recall_score

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_path',
    dest='model_path',
    type=str,
    required=True,
    help='path that the attack model is saved',
)

parser.add_argument(
    '--image_name',
    dest='image_name',
    type=str,
    required=True,
    help='path that the embeddings are stored: e.g.: ./res/model.pkl',
)

parser.add_argument(
    '--victim_feature',
    dest='victim_feature',
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


class attack_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.soft(out)
        return out



test_inputs = pd.read_csv(
    args.victim_feature, index_col=0)
test_labels = pd.read_csv(
    args.victim_label, index_col=0)

test_inputs = test_inputs.to_numpy()
test_inputs = test_inputs[:,feature_list]
test_labels = test_labels.to_numpy().reshape(test_inputs.shape[0],)


attack_testset = torch.utils.data.TensorDataset(torch.Tensor(
    test_inputs), torch.Tensor(test_labels))

attack_testloader = torch.utils.data.DataLoader(attack_testset, batch_size=32,
                                                shuffle=False, num_workers=0)

attack_model = attack_nn().cuda()
attack_model.load_state_dict(torch.load(args.model_path))


# evaluation
attack_model.eval()


predicted_labels = []
true_labels = []
test_correct = 0
test_total = 0
with torch.no_grad():
    for class_data in attack_testloader:
        class_images, class_labels = class_data
        class_outputs = attack_model(class_images.to(device))
        _, class_predicted = torch.max(class_outputs.data, 1)
        predicted_labels.extend(class_predicted.tolist())
        true_labels.extend(class_labels.tolist())
        test_total += class_labels.size(0)
        test_correct += (class_predicted ==
                         class_labels.to(device)).sum().item()

print(test_total)
print('Test Accuracy(net): %f %%' % (
    100 * test_correct / float(test_total)))


attack_outputs_test = np.array(predicted_labels)
true_labels = np.array(true_labels)
test_accuracy = accuracy_score(true_labels,attack_outputs_test)
test_precision = precision_score(true_labels,attack_outputs_test)
test_recall = recall_score(true_labels,attack_outputs_test)

print('Test Accuracy(net): %f %%' % (test_accuracy))
print("precision", test_precision)
print("recall",test_recall)