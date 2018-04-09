from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle
import time
import gc

data_perc = 1.0
use_validation = True

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

if use_validation:
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    del X, y
    gc.collect()
    print('{:.2f}s to split data in train/test'.format(time.time()-start))

model = RandomForestClassifier()

start = time.time()
if use_validation:
    model.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()
    print('{:.2f}s to train model'.format(time.time()-start))
    
    y_model = model.predict_proba(X_test)
    dvalid = xgb.DMatrix(X_test, y_test)
    print('roc auc score: {:.4f}'.format(roc_auc_score(y_model, y_test)))
    del X_test, y_test, y_model
    gc.collect()
else:
    model.fit(X, y)
    del X, y
    gc.collect()
    print('{:.2f}s to train model'.format(time.time()-start))

start = time.time()
pickle.dump(model, open("intermediary/model.rfc", "wb"))
print('{:.2f}s to save model to hd'.format(time.time()-start))

print('{:.2f}s to train model'.format(time.time()-raw_start))