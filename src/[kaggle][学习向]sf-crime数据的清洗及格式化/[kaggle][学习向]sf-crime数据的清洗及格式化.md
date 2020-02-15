# [kaggle][学习向]sf-crime数据的清洗及格式化

在上一篇[kaggle][学习向]sf-crime数据的可视化分析中，我们选取出了12个特征，但是数据仍然存放在csv表格中，我们将在这一篇中将数据进行清洗、格式化并存储成numpy易读取的模式，便于下一步的处理，下面开始编写程序，如果有不明确的地方，请参考上一篇

导入所需要使用的库：


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

读取train.csv和test.csv两个表格：


```python
train_data = pd.read_csv('../input/sf-crime/train.csv.zip', parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip', parse_dates=['Dates'])
```

将两个表格进行拼接，并抛弃训练集的Descript、Resolution两列，测试集的Id一列：


```python
all_features = pd.concat((train_data.iloc[:, [0, 3, 4, 6, 7, 8]],
                          test_data.iloc[:, [1, 2, 3, 4, 5, 6]]),
                         sort=False)
all_features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1762311 entries, 0 to 884261
    Data columns (total 6 columns):
    Dates         datetime64[ns]
    DayOfWeek     object
    PdDistrict    object
    Address       object
    X             float64
    Y             float64
    dtypes: datetime64[ns](1), float64(2), object(3)
    memory usage: 94.1+ MB
    

记录下训练集的行数：


```python
num_train = train_data.shape[0]
print(num_train)
```

    878049
    

生成训练集标签train_labels：


```python
train_labels = pd.get_dummies(train_data['Category']).values
num_outputs = train_labels.shape[1]
train_labels = np.argmax(train_labels, axis=1)
#PS：如果是使用Keras框架的categorical_crossentropy损失函数就可以不使用np.argmax
print(train_labels.shape)
print(train_labels[:10])
```

    (878049,)
    [37 21 21 16 16 16 36 36 16 16]
    

Dates列的数据包含年份、月份、新年（是否是1月、2月）、天、小时、黑夜（是否是18点之后）六个特征，分别进行处理：


```python
all_features['year'] = all_features.Dates.dt.year
all_features['month'] = all_features.Dates.dt.month
all_features['new_year'] = all_features['month'].apply(
    lambda x: 1 if x == 1 or x == 2 else 0)
all_features['day'] = all_features.Dates.dt.day
all_features['hour'] = all_features.Dates.dt.hour
all_features['evening'] = all_features['hour'].apply(lambda x: 1
                                                     if x >= 18 else 0)
```

处理DayOfWeek列数据，得到星期几和周末（是否是周五、周六）：


```python
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
```

独热编码（One-Hot Encoding），又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。即，只有一位是1，其余都是零值。

PdDistrict包含辖区的数据，我们选择用独热编码的方法进行处理：


```python
OneHot_features = pd.get_dummies(all_features['PdDistrict'])
```

提取出Address列中街区（是否存在block）的特征：


```python
all_features['block'] = all_features['Address'].apply(
    lambda x: 1 if 'block' in x.lower() else 0)
```

按照使用算法的区别，将all_features一分为三：


```python
PCA_features = all_features[['X', 'Y']].values
Standard_features = all_features[['DayOfWeek', 'year', 'month', 'day',
                                  'hour']].values
OneHot_features = pd.concat([
    OneHot_features, all_features[['new_year', 'evening', 'weekend', 'block']]
],
                            axis=1).values
```

特征缩放是用来统一资料中的自变项或特征范围的方法，我们采用特征缩放中标准化的方法对DayOfWeek、year、month、day、hour五列进行处理：


```python
scaler = StandardScaler()
scaler.fit(Standard_features)
Standard_features = scaler.transform(Standard_features)
```

主成分分析（Principal Component Analysis，PCA）， 是一种统计方法。通过正交变换将一组可能存在相关性的变量转换为一组线性不相关的变量。

我们既想保留X、Y的特征，又想适当削弱X、Y的权重，可以选择对X、Y两列进行主成分分析：


```python
pca = PCA(n_components=2)
pca.fit(PCA_features)
PCA_features = pca.transform(PCA_features)
```

将独热编码的PdDistrict、new_year、evening、weekend、block五部分与其余部分进行拼接，重新得到all_features（总计12个特征，21列）：


```python
all_features = np.concatenate(
    (PCA_features, Standard_features, OneHot_features), axis=1)
```

将all_features一分为二，得到处理好的训练集特征train_features和测试集特征test_features，以及网络输入层节点数num_inputs：


```python
train_features = all_features[:num_train]
num_inputs = train_features.shape[1]
test_features = all_features[num_train:]
```

查看训练集特征train_features、训练集标签train_labels、网络输入层节点数num_inputs、测试集特征test_features、网络输出层节点数num_outputs：


```python
np.save("./data/train_features.npy", train_features)
print('训练集特征储存于src/data文件夹下train_features')
print('train_features共%d行，%d列' %
      (train_features.shape[0], train_features.shape[1]))
np.save("./data/train_labels.npy", train_labels)
print('训练集标签储存于src/data文件夹下train_labels')
print('train_labels共%d行' % (train_labels.shape[0]))
np.save("./data/test_features.npy", (test_features))
print('测试集特征储存于src/data文件夹下test_features')
print('test_features共%d行，%d列' %
      (test_features.shape[0], test_features.shape[1]))
print('输入层节点数为%d' % (num_inputs))
print('输出层节点数为%d' % (num_outputs))
```

    训练集特征储存于src/data文件夹下train_features
    train_features共878049行，21列
    训练集标签储存于src/data文件夹下train_labels
    train_labels共878049行
    测试集特征储存于src/data文件夹下test_features
    test_features共884262行，21列
    输入层节点数为21
    输出层节点数为39
    

可以看出，训练集特征共21列（11个特征各1列，辖区独热编码占10列），878049行，即：有878049个样本供训练。测试集有884262个样本，需要计算这些样本39种犯罪类型各种类型的可能性（39种总计100%）。

至此，模型相关的数据处理基本完成，可以开始后面的步骤。
