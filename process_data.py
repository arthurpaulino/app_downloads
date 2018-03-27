from transform import transform
import pandas as pd
import numpy as np
import time
import gc

raw_start = time.time()

train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
    'click_id'      : 'uint32',
    'ip'            : 'uint32',
    'app'           : 'uint16',
	'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8'
}

# 0 < data_perc <= 1.0
data_perc = 0.2


# reading raw data
train_size_total = 184903890
train_size = int(data_perc*train_size_total)
start = time.time()
skiprows = range(1,train_size_total-train_size+1) if data_perc < 1.0 else None
print('reading input/train.csv')
data_train = pd.read_csv('input/train.csv', usecols=train_columns, dtype=dtypes, nrows=train_size, skiprows=skiprows)
print('{:.2f}s to load train data'.format(time.time()-start))

start = time.time()
print('reading input/test_supplement.csv')
data_test = pd.read_csv('input/test_supplement.csv', usecols=test_columns, dtype=dtypes)
print('{:.2f}s to load test data'.format(time.time()-start))


# extracting interesting features
def get_processed_data(data_train, data_test):
    start = time.time()
    train_size = data_train.shape[0]
    combine = pd.concat([data_train, data_test])
    combine['click_id'] = combine['click_id'].fillna(0).astype('uint8')
    combine['is_attributed'] = combine['is_attributed'].fillna(0).astype('uint8')
    del data_train, data_test
    gc.collect()
    print('{:.2f}s to concatenate train/test data'.format(time.time()-start))

    transform(combine)

    data_train = combine[:train_size].drop(columns=['click_id', 'ip', 'click_time'])
    data_test = combine[train_size:].drop(columns=['is_attributed'])

    del combine
    gc.collect()

    return (data_train, data_test)

start = time.time()
data_train, data_test = get_processed_data(data_train, data_test)
process_time = time.time()-start
print('{:.2f}s to process data ({:.2f} lines/s)'.format(process_time, (data_train.shape[0]+data_test.shape[0])/process_time))

print(data_train.head())
print(data_test.head())

# saving csv
start = time.time()
data_train.to_csv('intermediary/train_processed.csv', index=False)
del data_train
gc.collect()
print('{:.2f}s to create intermediary/train_processed.csv'.format(time.time()-start))

start = time.time()
data_submission = pd.read_csv('input/test.csv')
data_test = pd.merge(data_submission, data_test, how='inner', on=['ip', 'app', 'device', 'os', 'channel', 'click_time']) \
              .drop(columns=['ip', 'click_time', 'click_id_y']) \
              .drop_duplicates(subset=['click_id_x']) \
              .sort_values(by=['click_id_x']) \
              .rename(columns={'click_id_x': 'click_id'})
del data_submission
gc.collect()
print('{:.2f}s to choose submission subset'.format(time.time()-start))

print(data_test.head())

start = time.time()
data_test.to_csv('intermediary/test_processed.csv', index=False)
del data_test
gc.collect()
print('{:.2f}s to create intermediary/test_processed.csv'.format(time.time()-start))

print('{:.2f}s to run script'.format(time.time()-raw_start))
