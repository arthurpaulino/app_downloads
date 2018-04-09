import xgboost as xgb
import pandas as pd
import pickle
import time
import gc

raw_start = time.time()

dtypes = {}
with open('intermediary/test_processed.csv') as f:
    first_line = f.readline()
    features = first_line.split(',')
    for feature in features:
        if feature in ['app', 'os', 'device', 'channel', 'moment']:
            dtypes[feature] = 'uint16'
        else:
            dtypes[feature] = 'uint32'

start = time.time()
data_test = pd.read_csv('intermediary/test_processed.csv')
print('{:.2f}s to load test data'.format(time.time()-start))

submission = pd.DataFrame()
submission['click_id'] = data_test['click_id'].astype(int)

data_train = pd.read_csv('intermediary/train_processed.csv', nrows=1).drop(columns=['is_attributed'])
data_test = data_test[data_train.columns]
del data_train
gc.collect()

start = time.time()
dtest = xgb.DMatrix(data_test)
del data_test
gc.collect()
print('{:.2f}s to create xgboost data structure'.format(time.time()-start))

model = pickle.load(open("intermediary/model.xgb", "rb"))

start = time.time()
submission['is_attributed'] = model.predict(dtest)
del dtest
gc.collect()
print('{:.2f}s to make predictions'.format(time.time()-start))

start = time.time()
submission.to_csv('output/submission_xgb_'+str(int(time.time()))+'.csv', index=False)
print('{:.2f}s to write submission'.format(time.time()-start))

print('{:.2f}s to make predictions'.format(time.time()-raw_start))
