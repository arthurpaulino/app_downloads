from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import pickle
import time
import gc

data_perc = 1.0
use_gpu = False
use_validation = False

raw_start = time.time()

dtypes = {}
with open('intermediary/train_processed.csv') as f:
    first_line = f.readline()
    features = first_line.split(',')
    for feature in features:
        if feature=='is_attributed':
            dtypes[feature] = 'uint8'
        elif feature in ['app', 'os', 'device', 'channel', 'moment']:
            dtypes[feature] = 'uint16'
        else:
            dtypes[feature] = 'uint32'

if data_perc < 1.0:
    start = time.time()
    train_size_total = sum(1 for line in open('intermediary/train_processed.csv'))-1
    print('{:.2f}s to count training rows'.format(time.time()-start))

start = time.time()
if data_perc < 1.0:
    train_size = int(data_perc*train_size_total)
    data_train = pd.read_csv('intermediary/train_processed.csv', dtype=dtypes, nrows=train_size)
else:
    data_train = pd.read_csv('intermediary/train_processed.csv', dtype=dtypes)
print('{:.2f}s to load train data'.format(time.time()-start))

X = data_train.drop(columns=['is_attributed'])
y = data_train['is_attributed']

start = time.time()
unbalance_factor = data_train[data_train['is_attributed']==0].shape[0]/data_train[data_train['is_attributed']==1].shape[0]
del data_train
gc.collect()
print('{:.2f}s to compute unbalance factor: {}'.format(time.time()-start, unbalance_factor))

if use_validation:
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
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
    'min_child_weight':0,
    'alpha': 4,
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
if use_validation:
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
    model = xgb.train(xgb_params, dtrain, 200, watchlist, early_stopping_rounds = 25, verbose_eval=5)
    del dtrain, dvalid
    gc.collect()
    print('{:.2f}s to perform training'.format(time.time()-start))
else:
    dtrain = xgb.DMatrix(X, y)
    del X, y
    gc.collect()
    print('{:.2f}s to create xgboost data structures'.format(time.time()-start))

    # watch accuracy in train set
    watchlist = [(dtrain, 'training')]

    start = time.time()
    model = xgb.train(xgb_params, dtrain, 25, watchlist, verbose_eval=1)
    del dtrain
    gc.collect()
    print('{:.2f}s to perform training'.format(time.time()-start))

start = time.time()
_, ax = plt.subplots(figsize=(15,10))
xgb.plot_importance(model, ax=ax)
plt.savefig('intermediary/model_'+str(int(time.time()))+'.png')
print('{:.2f}s to compute fscores'.format(time.time()-start))

start = time.time()
pickle.dump(model, open("intermediary/model.xgb", "wb"))
print('{:.2f}s to save model to hd'.format(time.time()-start))

print('{:.2f}s to train model'.format(time.time()-raw_start))
