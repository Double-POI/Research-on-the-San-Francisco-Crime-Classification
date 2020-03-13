import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

parameters = {
    'max_depth': [10, 30, 50, 70, 90],
}
model = RandomForestClassifier(n_estimators=50, max_depth=10)
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
{'max_depth': [10, 30, 50, 70, 90]}
[-2.45267495 -5.63315795 -6.73742927 -6.74492684 -6.74814227]
[0.00077951 0.02050809 0.01977714 0.0103057  0.00515421]
{'max_depth': 10}
-2.45267495266959
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

parameters = {
    'max_depth': [5,6,7,8,9,10],
}
model = RandomForestClassifier(n_estimators=50, max_depth=10)
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
{'max_depth': [5, 6, 7, 8, 9, 10]}
[-2.54637367 -2.52371166 -2.50549146 -2.48602735 -2.46992585 -2.45302022]
[0.00167687 0.00274124 0.00204834 0.00095764 0.00031478 0.00141425]
{'max_depth': 10}
-2.4530202156447216
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train_features_raw = np.load('./data/train_features.npy')
train_labels_raw = np.load('./data/train_labels.npy')

index = [i for i in range(len(train_features_raw))]
np.random.shuffle(index)
train_features = train_features_raw[index]
train_labels = train_labels_raw[index]

parameters = {
    'min_samples_split': [50, 100, 150],
}
model = RandomForestClassifier(n_estimators=50, max_depth=10)
gsearch = GridSearchCV(model,
                       param_grid=parameters,
                       scoring='neg_log_loss',
                       cv=3)
gsearch.fit(train_features, train_labels)
print(gsearch.cv_results_['params'])
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['std_test_score'])
print(gsearch.best_params_)
print(gsearch.best_score_)

'''
[{'min_samples_split': 50}, {'min_samples_split': 100}, {'min_samples_split': 150}]
[-2.45253602 -2.45440453 -2.45478853]
[0.00062034 0.00112358 0.0003685 ]
{'min_samples_split': 50}
-2.452536016669535
'''