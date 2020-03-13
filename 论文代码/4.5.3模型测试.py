import pandas as pd
import numpy as np
import lightgbm as lgb

train_features = np.load('./data/train_features.npy')
train_labels = np.load('./data/train_labels.npy')
test_features = np.load('./data/test_features.npy')

num_inputs = 21
num_outputs = 39

data_train = lgb.Dataset(train_features, label=train_labels)

params = {
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'metrics': 'multi_logloss',
    'num_class': num_outputs,
    'verbosity': 1,
    'max_depth': 6,
    'num_leaves': 51,
    'min_data_in_leaf': 25,
    'feature_fraction': 0.79,
    'learning_rate': 0.01,
}
gbm = lgb.train(params, data_train, num_boost_round=2000)
gbm.save_model('../working/gbm.txt')
testResult = gbm.predict(test_features)
sampleSubmission = pd.read_csv('../input/sf-crime/sampleSubmission.csv.zip')
Result_pd = pd.DataFrame(testResult,
                         index=sampleSubmission.index,
                         columns=sampleSubmission.columns[1:])
Result_pd.to_csv('../working/sampleSubmission(gbm).csv', index_label='Id')
