import pandas as pd
import numpy as np
import torch
from torch import nn

train_features = np.load('./data/train_features.npy')
train_labels = np.load('./data/train_labels.npy')
train_features = torch.tensor(train_features, dtype=torch.float)
train_labels = torch.tensor(train_labels)

num_inputs = 21
num_outputs = 39

num_epochs = 5
batch_size = 32
lr = 0.1

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
net.add_module('Dense', nn.Linear(num_inputs, num_outputs))

loss = nn.CrossEntropyLoss()
# torch框架中CrossEntropyLoss自带Softmax运算，所以网络部分没有Softmax层

optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    print('epoch %d, loss %.4f, train acc %.3f' %
          (epoch + 1, train_l_sum / n, train_acc_sum / n))

'''
epoch 1, loss 0.0802, train acc 0.229
epoch 2, loss 0.0797, train acc 0.230
epoch 3, loss 0.0796, train acc 0.230
epoch 4, loss 0.0796, train acc 0.229
epoch 5, loss 0.0796, train acc 0.230
'''
