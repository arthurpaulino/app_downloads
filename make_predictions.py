import xgboost as xgb
import pandas as pd
import pickle
import time
import gc

raw_start = time.time()

start = time.time()
data_test = pd.read_csv('input/test_processed.csv')
print('{:.2f}s to read test data'.format(time.time()-start))

submission = pd.DataFrame()
submission['click_id'] = data_test['click_id'].astype(int)

start = time.time()
dtest = xgb.DMatrix(data_test.drop(columns=['click_id']))
del data_test
gc.collect()
print('{:.2f}s to create xgboost data structure'.format(time.time()-start))

model = pickle.load(open("intermediary/model.xgb", "rb"))

start = time.time()
submission['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
del dtest
gc.collect()
print('{:.2f}s to make predictions'.format(time.time()-start))

start = time.time()
submission.to_csv('output/submission_'+str(int(time.time()))+'.csv', index=False)
print('{:.2f}s to write submission'.format(time.time()-start))

print('{:.2f}s to run script'.format(time.time()-raw_start))
