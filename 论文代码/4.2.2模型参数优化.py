import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

parameters = {
    'alpha': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
model = BernoulliNB(alpha=1.0)
gsearch = GridSearchCV(model,
                       param_grid=parameters,
                       scoring='neg_log_loss',
                       cv=3)
gsearch.fit(train_features, train_labels)
print(parameters)
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['std_test_score'])
print(gsearch.best_params_)
print(gsearch.best_score_)

'''
{'alpha': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
[-2.58388939 -2.58387524 -2.58386181 -2.58384895 -2.58383655 -2.58382453]
[0.00151963 0.00152036 0.00152083 0.00152111 0.00152125 0.00152128]
{'alpha': 1.0}
-2.5838245319029713
'''