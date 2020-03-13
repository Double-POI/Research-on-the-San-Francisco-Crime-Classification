results = lgb.cv(params,
                 data_train,
                 num_boost_round = 1000,
                 nfold = 3,
                 shuffle = True,
                 early_stopping_rounds = 40,
                 verbose_eval = 100)
print('best num_boost_round:', len(results['multi_logloss-mean']))
print('last mean:', results['multi_logloss-mean'][-1])
print('last stdv:', results['multi_logloss-stdv'][-1])

'''
[100] cv_agg’s multi_logloss: 2.39575 + 0.00471342
[200] cv_agg’s multi_logloss: 2.3862 + 0.00131608
best num_boost_round: 214
last mean: 2.379645842199627
last stdv: 0.0024214570330422813
'''