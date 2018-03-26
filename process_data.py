from transform import transform
import pandas as pd
import numpy as np
import time
import gc

raw_start = time.time()

train_columns = ['ip', 'app', 'os', 'device', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'os', 'device', 'channel', 'click_time', 'click_id']
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'os'            : 'uint16',
    'device'        : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

# 0 < data_perc <= 1.0
data_perc = 0.3


# reading raw data
train_size_total = 184903890
train_size = int(data_perc*train_size_total)
start = time.time()
skiprows = range(1,train_size_total-train_size+1) if data_perc < 1.0 else None
print('reading input/train.csv')
data_train = pd.read_csv('input/train.csv', usecols=train_columns, dtype=dtypes, parse_dates=['click_time'], nrows=train_size, skiprows=skiprows)
print('{:.2f}s to load train data'.format(time.time()-start))

start = time.time()
print('reading input/test.csv')
data_test = pd.read_csv('input/test.csv', usecols=test_columns, dtype=dtypes, parse_dates=['click_time'])
print('{:.2f}s to load test data'.format(time.time()-start))


# extracting interesting features
def get_processed_data(data_train, data_test):
    start = time.time()
    train_size = data_train.shape[0]
    combine = pd.concat([data_train, data_test])
    combine['click_id'] = combine['click_id'].fillna(0).astype('uint32')
    combine['is_attributed'] = combine['is_attributed'].fillna(0).astype('uint8')
    del data_train, data_test
    gc.collect()
    print('{:.2f}s to concatenate train/test data'.format(time.time()-start))

    transform(combine)

    data_train = combine[:train_size].drop(columns=['click_id'])
    data_test = combine[train_size:].drop(columns=['is_attributed'])

    del combine
    gc.collect()

    return (data_train, data_test)

start = time.time()
data_train, data_test = get_processed_data(data_train, data_test)
process_time = time.time()-start
print('{:.2f}s to process data ({:.2f} lines/s)'.format(process_time, (data_train.shape[0]+data_test.shape[0])/process_time))


# saving csv
start = time.time()
data_train.to_csv('input/train_processed.csv', index=False)
del data_train
gc.collect()
print('{:.2f}s to create input/train_processed.csv'.format(time.time()-start))

start = time.time()
data_test.to_csv('input/test_processed.csv', index=False)
del data_test
gc.collect()
print('{:.2f}s to create input/test_processed.csv'.format(time.time()-start))

print('{:.2f}s to run script'.format(time.time()-raw_start))
