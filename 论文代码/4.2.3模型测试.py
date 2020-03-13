import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')
test_features = np.load('./data/test_features.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

model = BernoulliNB(alpha=1.0)
model.fit(train_features, train_labels)
predicted = np.array(model.predict_proba(train_features))
print("朴素贝叶斯的训练log损失为 %f" % (log_loss(train_labels, predicted)))
testResult = np.array(model.predict_proba(test_features))
sampleSubmission = pd.read_csv('../input/sf-crime/sampleSubmission.csv.zip')
Result_pd = pd.DataFrame(testResult,
                         index=sampleSubmission.index,
                         columns=sampleSubmission.columns[1:])
Result_pd.to_csv('../working/sampleSubmission(bayes).csv', index_label='Id')

#朴素贝叶斯的训练log损失为 2.582578