#作者：1621430024

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#train_data = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip',
#                         parse_dates=['Dates'])
train_data = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
#test_data = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip',
#                        parse_dates=['Dates'])
test_data = pd.read_csv('../input/test.csv', parse_dates=['Dates'])

all_features = pd.concat((train_data.iloc[:, [0, 3, 4, 6, 7, 8]],
                          test_data.iloc[:, [1, 2, 3, 4, 5, 6]]),
                         sort=False)

num_train = train_data.shape[0]

train_labels = pd.get_dummies(train_data['Category']).values
num_outputs = train_labels.shape[1]

all_features['year'] = all_features.Dates.dt.year
all_features['month'] = all_features.Dates.dt.month
all_features['new_year'] = all_features['month'].apply(
    lambda x: 1 if x == 1 or x == 2 else 0)
all_features['day'] = all_features.Dates.dt.day
all_features['hour'] = all_features.Dates.dt.hour
all_features['evening'] = all_features['hour'].apply(lambda x: 1
                                                     if x >= 18 else 0)

wkm = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
all_features['DayOfWeek'] = all_features['DayOfWeek'].apply(lambda x: wkm[x])
all_features['weekend'] = all_features['DayOfWeek'].apply(
    lambda x: 1 if x == 4 or x == 5 else 0)

OneHot_features = pd.get_dummies(all_features['PdDistrict'])

all_features['block'] = all_features['Address'].apply(
    lambda x: 1 if 'block' in x.lower() else 0)

PCA_features = all_features[['X', 'Y']].values
Standard_features = all_features[['DayOfWeek', 'year', 'month', 'day',
                                  'hour']].values
OneHot_features = pd.concat([
    OneHot_features, all_features[['new_year', 'evening', 'weekend', 'block']]
],
                            axis=1).values

scaler = StandardScaler()
scaler.fit(Standard_features)
Standard_features = scaler.transform(Standard_features)

pca = PCA(n_components=2)
pca.fit(PCA_features)
PCA_features = pca.transform(PCA_features)

all_features = np.concatenate(
    (PCA_features, Standard_features, OneHot_features), axis=1)

train_features = all_features[:num_train]
num_inputs = train_features.shape[1]
test_features = all_features[num_train:]


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


net = build_model(num_inputs, num_outputs).cuda()


class MultiClassLogLoss(torch.nn.Module):
    def __init__(self):
        super(MultiClassLogLoss, self).__init__()
    def forward(self, y_pred, y_true):
        return -(y_true *
                 torch.log(y_pred.float() + 1.00000000e-15)) / y_true.shape[0]


loss = MultiClassLogLoss().cuda()


def make_iter(train_features, train_labels, batch_size):
    train_features = torch.tensor(train_features, dtype=torch.float).cuda()
    train_labels = torch.tensor(train_labels).cuda()
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
    #test_iter = torch.utils.data.DataLoader(torch.tensor(
    #    test_features, dtype=torch.float).cuda(),
    #                                        1024,
    #                                        shuffle=False)
    test_iter = torch.utils.data.DataLoader(torch.tensor(
        test_features, dtype=torch.float).cuda(),
                                            1024,
                                            shuffle=False)
    #testResult = [
    #    line for x in test_iter for line in net(x).cpu().detach().numpy()
    #]
    testResult = [
        line for x in test_iter for line in net(x).cpu().detach().numpy()
    ]
    #sampleSubmission = pd.read_csv(
    #    '/kaggle/input/sf-crime/sampleSubmission.csv.zip')
    sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
    Result_pd = pd.DataFrame(testResult,
                             index=sampleSubmission.index,
                             columns=sampleSubmission.columns[1:])
    #Result_pd.to_csv('/kaggle/working/sampleSubmission(v0.5).csv',
    #                 index_label='Id')
    Result_pd.to_csv('../output/sampleSubmission(v0.5).csv', index_label='Id')