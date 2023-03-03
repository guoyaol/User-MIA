from cProfile import label
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
parser.add_argument("--epochs",
                    default=3000,
                    type=int,
                    help="traing epoch")
parser.add_argument("--lr",
                    default=1e-4,
                    type=float,
                    help="learning rate")

parser.add_argument(
    '--attack_name',
    dest='attack_name',
    type=str,
    default='noname',
    help='path that the embeddings are stored: e.g.: ./res/model.pkl',
)

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


attack_trainset = torch.utils.data.TensorDataset(torch.Tensor(
    train_inputs), torch.Tensor(train_labels))

attack_trainloader = torch.utils.data.DataLoader(attack_trainset, batch_size=32,
                                                 shuffle=True, num_workers=0)

attack_testset = torch.utils.data.TensorDataset(torch.Tensor(
    test_inputs), torch.Tensor(test_labels))

attack_testloader = torch.utils.data.DataLoader(attack_testset, batch_size=32,
                                                shuffle=True, num_workers=0)

attack_model = attack_nn()
attack_criterion = nn.CrossEntropyLoss()
attack_optimizer = optim.Adam(attack_model.parameters(), lr=args.lr)

attack_model.to(device)
attack_model.train()
for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(attack_trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        attack_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = attack_model(inputs.to(device))
        # batch_hard_triplet_loss(labels=labels,embeddings=outputs,margin=0.3)
        loss = attack_criterion(
            outputs, labels.to(device, dtype=torch.long))
        loss.backward()
        attack_optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('%d, loss: %.3f' %
          (epoch + 1, running_loss / len(attack_trainloader)))

# evaluation
attack_model.eval()

attack_outputs_train = attack_model(torch.tensor(
    train_inputs).float().to(device))
attack_outputs_train = attack_outputs_train.detach().cpu().numpy()
single_conf_train = attack_outputs_train[..., 1]
real_labels_train = train_labels.astype(int)
train_auc_score = roc_auc_score(
    real_labels_train, single_conf_train)

attack_outputs_test = attack_model(torch.tensor(
    test_inputs).float().to(device))
attack_outputs_test = attack_outputs_test.detach().cpu().numpy()
single_conf_test = attack_outputs_test[..., 1]
real_labels_test = test_labels.astype(int)
test_auc_score = roc_auc_score(
    real_labels_test, single_conf_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(real_labels_test, single_conf_test)
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("./saved_pics/"+args.attack_name)


train_correct = 0
train_total = 0
with torch.no_grad():
    for class_data in attack_trainloader:
        class_images, class_labels = class_data
        class_outputs = attack_model(class_images.to(device))
        _, class_predicted = torch.max(class_outputs.data, 1)
        train_total += class_labels.size(0)
        train_correct += (class_predicted ==
                          class_labels.to(device)).sum().item()

print(train_total)
print('Train accuracy(net): %f %%' % (
    100 * train_correct / float(train_total)))

test_correct = 0
test_total = 0
predicted_labels = []
true_labels = []
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

save_path = './attack_model_new/'+args.attack_name+'.pth'
torch.save(attack_model.state_dict(), save_path)

attack_outputs_test = np.array(predicted_labels)
true_labels = np.array(true_labels)
test_accuracy = accuracy_score(true_labels,attack_outputs_test)
test_precision = precision_score(true_labels,attack_outputs_test)
test_recall = recall_score(true_labels,attack_outputs_test)

print('Test Accuracy(net): %f %%' % (test_accuracy))
print("precision", test_precision)
print("recall",test_recall)