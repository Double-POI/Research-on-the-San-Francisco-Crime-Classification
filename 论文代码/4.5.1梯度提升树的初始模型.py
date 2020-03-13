import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

train_features = np.load('./data/train_features.npy')
train_labels = np.load('./data/train_labels.npy')
test_features = np.load('./data/test_features.npy')

data_train = lgb.Dataset(train_features, label=train_labels)

num_inputs = 21
num_outputs = 39

params = {
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'metrics': 'multi_logloss',
    'num_class': num_outputs,
    'verbosity': 1,
    'max_depth': 6,
    'num_leaves': 50,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'learning_rate': 0.1,
}

gbm = lgb.train(params, data_train, num_boost_round=200)
predicted = gbm.predict(train_features)
print("梯度提升树的训练log损失为 %f" % (log_loss(train_labels, predicted)))

#梯度提升树的训练log损失为 2.384802