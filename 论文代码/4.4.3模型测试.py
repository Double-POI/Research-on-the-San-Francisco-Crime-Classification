import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold

train_features = np.load('./data/train_features.npy')
train_labels = np.load('./data/train_labels_onehot.npy')
test_features = np.load('./data/test_features.npy')

num_inputs = 21
num_outputs = 39


class Residual(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Residual, self).__init__()
        self.middle_L = nn.Linear(num_inputs, num_outputs)
        self.middle_R = nn.ReLU(num_outputs)
        if num_inputs != num_outputs:
            self.right = nn.Linear(num_inputs, num_outputs)
        else:
            self.right = None
        self.middle_B = nn.BatchNorm1d(num_outputs)

    def forward(self, X):
        Y = self.middle_B(self.middle_R(self.middle_L(X)))
        if self.right:
            X = self.right(X)
        return Y + X


class build_model(nn.Module):
    def __init__(self, num_inputs, num_outputs, dp=0.5):
        super(build_model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('Residual1', Residual(num_inputs, 1024))
        self.net.add_module('Residual2', Residual(1024, 512))
        self.net.add_module('Residual3', Residual(512, 512))
        self.net.add_module('Residual4', Residual(512, 256))
        self.net.add_module('Dropout1', nn.Dropout(dp))
        self.net.add_module('Residual5', Residual(256, 256))
        self.net.add_module('Residual6', Residual(256, 128))
        self.net.add_module('Residual7', Residual(128, 128))
        self.net.add_module('Residual8', Residual(128, 64))
        self.net.add_module('Dropout2', nn.Dropout(dp))
        self.net.add_module('Residual9', Residual(64, 64))
        self.net.add_module('Linear-out', nn.Linear(64, num_outputs))
        self.net.add_module('Softmax', nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


net = build_model(num_inputs, num_outputs)


class MultiClassLogLoss(torch.nn.Module):
    def __init__(self):
        super(MultiClassLogLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -(y_true *
                 torch.log(y_pred.float() + 1.00000000e-15)) / y_true.shape[0]


loss = MultiClassLogLoss()


def make_iter(train_features, train_labels, batch_size):
    train_features = torch.tensor(train_features, dtype=torch.float)
    train_labels = torch.tensor(train_labels)
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def show_loss(net, loss, features, labels, team):
    net.eval()
    batch = make_iter(features, labels, 1024)
    loss_num = 0
    n = 0
    for x, y in batch:
        loss_num += loss(net(x), y).sum().item()
        n += 1
    print(team, end=' ')
    print('loss:', loss_num / n)


def train(features, labels, batch_size):
    net.train()
    train_iter = make_iter(features, labels, batch_size)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    show_loss(net, loss, features, labels, '训练集')


#num_epochs = 10
num_epochs = 100
k_fold_num = 5
batch_size = 128
lr = 0.001
#k_fold = True
k_fold = False

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

if k_fold:
    kf = KFold(n_splits=k_fold_num, shuffle=True)
    for epoch in range(num_epochs):
        fold_num = 0
        for train_index, test_index in kf.split(train_features):
            X_train, X_test = train_features[train_index], train_features[
                test_index]
            y_train, y_test = train_labels[train_index], train_labels[
                test_index]
            print('第%d轮的第%d折：' % (epoch + 1, fold_num + 1))
            fold_num += 1
            train(X_train, y_train, batch_size)
            show_loss(net, loss, X_test, y_test, '测试集')
else:
    for epoch in range(num_epochs):
        print('第%d轮：' % (epoch + 1))
        train(train_features, train_labels, batch_size)

    net.eval()
    test_iter = torch.utils.data.DataLoader(torch.tensor(test_features,
                                                         dtype=torch.float),
                                            1024,
                                            shuffle=False)
    testResult = [line for x in test_iter for line in net(x).detach().numpy()]
    sampleSubmission = pd.read_csv(
        '../input/sf-crime/sampleSubmission.csv.zip')
    Result_pd = pd.DataFrame(testResult,
                             index=sampleSubmission.index,
                             columns=sampleSubmission.columns[1:])
    Result_pd.to_csv('../working/sampleSubmission(NN).csv', index_label='Id')
    torch.save(net, '../working/net.pkl')
