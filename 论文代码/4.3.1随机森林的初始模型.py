import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(train_features, train_labels)
predicted = np.array(model.predict_proba(train_features))
print("随机森林的训练log损失为 %f" % (log_loss(train_labels, predicted)))

#随机森林的训练log损失为 2.544871