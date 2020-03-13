import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

train_data = pd.read_csv('../input/sf-crime/train.csv.zip',
                         parse_dates=['Dates'])
test_data = pd.read_csv('../input/sf-crime/test.csv.zip',
                        parse_dates=['Dates'])
#读取数据
all_features = pd.concat((train_data.iloc[:, [0, 3, 4, 6, 7, 8]],
                          test_data.iloc[:, [1, 2, 3, 4, 5, 6]]),
                         sort=False)
#抛弃Descript、Resolution、Id列
num_train = train_data.shape[0]
#记录num_train
train_labels = pd.get_dummies(train_data['Category']).values
np.save("data/train_labels_onehot.npy", train_labels)
num_outputs = train_labels.shape[1]
train_labels = np.argmax(train_labels, axis=1)
np.save("data/train_labels.npy", train_labels)
#生成train_labels
all_features['year'] = all_features.Dates.dt.year
all_features['month'] = all_features.Dates.dt.month
all_features['new_year'] = all_features['month'].apply(
    lambda x: 1 if x == 1 or x == 2 else 0)
all_features['day'] = all_features.Dates.dt.day
all_features['hour'] = all_features.Dates.dt.hour
all_features['evening'] = all_features['hour'].apply(lambda x: 1
                                                     if x >= 18 else 0)
#处理Dates列
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
#处理DayOfWeek列
OneHot_features = pd.get_dummies(all_features['PdDistrict'])
#处理PdDistrict列
all_features['block'] = all_features['Address'].apply(
    lambda x: 1 if 'block' in x.lower() else 0)
#处理Address列
PCA_features = all_features[['X', 'Y']].values
Standard_features = all_features[['DayOfWeek', 'year', 'month', 'day',
                                  'hour']].values
OneHot_features = pd.concat([
    OneHot_features, all_features[['new_year', 'evening', 'weekend', 'block']]
],
                            axis=1).values
#对'PdDistrict','new_year','evening','weekend','block'进行独热编码
#并将all_features按处理方式进行拆分
scaler = StandardScaler()
scaler.fit(Standard_features)
Standard_features = scaler.transform(Standard_features)
#对'DayOfWeek','year','month','day','hour'进行标准化
pca = PCA(n_components=2)
pca.fit(PCA_features)
PCA_features = pca.transform(PCA_features)
#对'X','Y'进行主成分分析
all_features = np.concatenate(
    (PCA_features, Standard_features, OneHot_features), axis=1)
#完成拼接
train_features = all_features[:num_train]
num_inputs = train_features.shape[1]
test_features = all_features[num_train:]
#至此，获得训练集的train_features(num_inputs)和train_labels(num_outputs)，测试集的test_features
np.save("data/train_features.npy", train_features)
print('训练集特征储存于src/data文件夹下train_features.npy')
print('train_features共%d行，%d列' %
      (train_features.shape[0], train_features.shape[1]))
print('训练集标签储存于src/data文件夹下train_labels.npy')
print('训练集（独热）标签储存于src/data文件夹下train_labels_onehot.npy')
print('train_labels共%d行' % (train_labels.shape[0]))
np.save("data/test_features.npy", (test_features))
print('测试集特征储存于src/data文件夹下test_features.npy')
print('test_features共%d行，%d列' %
      (test_features.shape[0], test_features.shape[1]))
print('输入节点数为%d' % (num_inputs))
print('输出节点数为%d' % (num_outputs))
