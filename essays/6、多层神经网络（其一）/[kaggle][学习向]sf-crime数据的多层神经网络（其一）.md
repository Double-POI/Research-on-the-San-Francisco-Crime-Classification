# [kaggle][学习向]sf-crime数据的多层神经网络（其一）

在上一篇[kaggle][学习向]sf-crime数据的单层神经网络中，我们完成了一个简单的神经网络，完成了第一次的训练、预测与提交，但是这不够，我们需要更好，这个比赛的数据更适合传统的机器学习方法，笔者在进行多次整理、加深、调参之后停留在了Top30%，（但是已经是这个比赛里面使用NN的notebook里面最好的成绩了），相信大家能够超越我，加油！

在上一篇中，我们的神经网络已经能够完成整个流程，那么我们应该怎样提升它呢？

一、需要注意到，此比赛使用的评分标准是log_loss函数，而我们使用的是torch的CrossEntropyLoss，这不适合我们评断我们的训练效果

二、kaggle比赛的测试集是没有标签的，也就是说，除了提交，我们没有一个很好的方式判断我们训练的效果

所以我们需要整理一次项目，为后面的调整做铺垫

让我们开始

导入所需要的库：


```python
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
```

导入处理好的训练集的特征和标签：


```python
train_features = np.load('./data/train_features.npy')
train_labels = np.load('./data/train_labels_log.npy')
test_features = np.load('./data/test_features.npy')

num_inputs = 21
num_outputs = 39
```

## 自定义损失loss函数

log_loss是在机器学习构建分类模型的任务中经常使用的损失度量方法，公式是：

让我们自定义一个log_loss：


```python
class MultiClassLogLoss(torch.nn.Module):
    def __init__(self):
        super(MultiClassLogLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -(y_true *
                 torch.log(y_pred.float() + 1.00000000e-15)) / y_true.shape[0]
# 防止出现log(0)，加1*10^-15

loss = MultiClassLogLoss()
```

## 定义模型


```python
class build_model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(build_model, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module('Linear', nn.Linear(num_inputs, num_outputs))
        self.net.add_module('Softmax', nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)
    
net = build_model(num_inputs, num_outputs)
```

## 批量读取数据函数


```python
def make_iter(train_features, train_labels, batch_size):
    train_features = torch.tensor(train_features, dtype=torch.float)
    train_labels = torch.tensor(train_labels)
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
```

## 训练/泛化误差计算函数


```python
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
```

## 训练函数


```python
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
```

## 设置超参数


```python
num_epochs = 5
k_fold_num = 5
batch_size = 32
lr = 0.1
k_fold = True
```

## 定义优化函数


```python
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
```

## 引入K折交叉验证

用同一数据集，既进行训练，又进行模型误差估计，对误差的估计会出现不准确的问题，这就是所谓的模型误差估计的乐观性。为了克服这个问题，我们引入了交叉验证。

基本思想是将数据分为两部分，一部分数据用来模型的训练，称为训练集；另外一部分用于测试模型的误差，称为验证集。

由于两部分数据不同，估计得到的泛化误差更接近真实的模型表现。能更好的评估模型的效果。

## 开始训练


```python
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
    sampleSubmission = pd.read_csv('../input/sf-crime/sampleSubmission.csv.zip')
    Result_pd = pd.DataFrame(testResult,
                             index=sampleSubmission.index,
                             columns=sampleSubmission.columns[1:])
    Result_pd.to_csv('../working/sampleSubmission(v0.1).csv', index_label='Id')
```

    第1轮的第1折：
    训练集 loss: 2.5503316954343034
    测试集 loss: 2.5525896840317306
    第1轮的第2折：
    训练集 loss: 2.549008633930551
    测试集 loss: 2.5492086715476456
    第1轮的第3折：
    训练集 loss: 2.5463152279311645
    测试集 loss: 2.5483997782995536
    第1轮的第4折：
    训练集 loss: 2.546247698475251
    测试集 loss: 2.548039274160252
    第1轮的第5折：
    训练集 loss: 2.548516502185744
    测试集 loss: 2.5459297632062157
    第2轮的第1折：
    训练集 loss: 2.5463628101626914
    测试集 loss: 2.548425505327624
    第2轮的第2折：
    训练集 loss: 2.546687583881634
    测试集 loss: 2.5528524781382362
    第2轮的第3折：
    训练集 loss: 2.5462778594681543
    测试集 loss: 2.543629068274831
    第2轮的第4折：
    训练集 loss: 2.54511651283798
    测试集 loss: 2.545532415079516
    第2轮的第5折：
    训练集 loss: 2.5477974574697955
    测试集 loss: 2.5463107075802114
    第3轮的第1折：
    训练集 loss: 2.5450612396957575
    测试集 loss: 2.5497251097546068
    第3轮的第2折：
    训练集 loss: 2.5462568018248755
    测试集 loss: 2.547771273657333
    第3轮的第3折：
    训练集 loss: 2.545838413363643
    测试集 loss: 2.544646130051724
    第3轮的第4折：
    训练集 loss: 2.545403643182693
    测试集 loss: 2.5437340140342712
    第3轮的第5折：
    训练集 loss: 2.5455294635483545
    测试集 loss: 2.5471859920856565
    第4轮的第1折：
    训练集 loss: 2.5473473933973394
    测试集 loss: 2.54685703682345
    第4轮的第2折：
    训练集 loss: 2.546726713375169
    测试集 loss: 2.544034934321115
    第4轮的第3折：
    训练集 loss: 2.5443586371730436
    测试集 loss: 2.548985499282216
    第4轮的第4折：
    训练集 loss: 2.545617711439772
    测试集 loss: 2.547614945921787
    第4轮的第5折：
    训练集 loss: 2.544828007242075
    测试集 loss: 2.544772464175557
    第5轮的第1折：
    训练集 loss: 2.545793165270858
    测试集 loss: 2.5451489035473314
    第5轮的第2折：
    训练集 loss: 2.5436720309382626
    测试集 loss: 2.5482053590375324
    第5轮的第3折：
    训练集 loss: 2.5440436987418127
    测试集 loss: 2.5448591612106144
    第5轮的第4折：
    训练集 loss: 2.5444287801970544
    测试集 loss: 2.5450654279354006
    第5轮的第5折：
    训练集 loss: 2.5461313164964015
    测试集 loss: 2.5441392119540724
    

观察训练过程中loss的变化，我们可以发现，训练误差和泛化误差差值不大，即暂没出现过拟合现象，但是训练集误差徘徊在2.545附近，还有进步空间，由于并未改变网络结构，我们暂时不得到输出
