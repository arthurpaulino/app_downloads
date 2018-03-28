from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import pickle
import time
import gc

raw_start = time.time()

data_perc = 1.0
use_gpu = False

if data_perc < 1.0:
    start = time.time()
    train_size_total = sum(1 for line in open('intermediary/train_processed.csv'))-1
    print('{:.2f}s to count training rows'.format(time.time()-start))

start = time.time()
if data_perc < 1.0:
    train_size = int(data_perc*train_size_total)
    data_train = pd.read_csv('intermediary/train_processed.csv', nrows=train_size)
else:
    data_train = pd.read_csv('intermediary/train_processed.csv')
print('{:.2f}s to load train data'.format(time.time()-start))

X = data_train.drop(columns=['is_attributed'])
y = data_train['is_attributed']

start = time.time()
unbalance_factor = data_train[data_train['is_attributed']==0].shape[0]/data_train[data_train['is_attributed']==1].shape[0]
del data_train
gc.collect()
print('{:.2f}s to compute unbalance factor: {}'.format(time.time()-start, unbalance_factor))

start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
del X, y
gc.collect()
print('{:.2f}s to split data in train/test'.format(time.time()-start))

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
xgb_params = {
    'eta': 0.1,
    'max_leaves': 1024,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'colsample_bylevel':0.7,
	'max_delta_step': 1,
    'min_child_weight':2,
    'alpha': 3,
	'gamma ': 1.2,
    'max_depth': 0,
    'scale_pos_weight': unbalance_factor,
    'eval_metric': 'auc',
    'random_state': int(time.time()),
    'silent': True,
    'grow_policy': 'lossguide',
    'tree_method': 'hist',
    'predictor': 'cpu_predictor',
    'objective': 'binary:logistic'
}

if use_gpu:
    xgb_params.update({'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'objective':'gpu:binary:logistic'})

start = time.time()
dtrain = xgb.DMatrix(X_train, y_train)
del X_train, y_train
gc.collect()
dvalid = xgb.DMatrix(X_test, y_test)
del X_test, y_test
gc.collect()
print('{:.2f}s to create xgboost data structures'.format(time.time()-start))

# watch accuracy in validation set
watchlist = [(dvalid, 'validation')]

start = time.time()
model = xgb.train(xgb_params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
del dvalid, dtrain
gc.collect()
print('{:.2f}s to perform training'.format(time.time()-start))

start = time.time()
_, ax = plt.subplots(figsize=(15,5))
xgb.plot_importance(model, ax=ax)
plt.savefig('intermediary/model_'+str(int(time.time()))+'.png')
print('{:.2f}s to save feature importance plot'.format(time.time()-start))

start = time.time()
pickle.dump(model, open("intermediary/model.xgb", "wb"))
print('{:.2f}s to save model to hd'.format(time.time()-start))

print('{:.2f}s to run script'.format(time.time()-raw_start))
