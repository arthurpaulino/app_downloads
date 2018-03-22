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
data_perc = 1


# reading raw data

train_size_total = 184903890
train_size = int(data_perc*train_size_total)
start = time.time()
data_train = pd.read_csv('input/train.csv', usecols=train_columns, parse_dates=['click_time'], dtype=dtypes, nrows=train_size, skiprows=range(1,train_size_total-train_size+1))
print('{:.2f}s to load train data'.format(time.time()-start))

start = time.time()
data_test = pd.read_csv('input/test.csv', usecols=test_columns, parse_dates=['click_time'], dtype=dtypes)
print('{:.2f}s to load test data'.format(time.time()-start))


# extracting interesting features

def get_attributed(obj):
    sm = obj.sum()
    if np.isnan(sm):
        return 0
    return sm

def replace_base_features_by_their_perc(combine, features):
    for feature in features:
        start = time.time()
        total = combine.groupby(feature)[feature].transform('count').astype('uint32')
        attributed = combine.groupby(feature)['is_attributed'].transform(get_attributed)
        combine[feature] = attributed/total
        del total, attributed
        gc.collect()
        print('{:.2f}s to remake feature {}'.format(time.time()-start, feature))

def get_timediff(obj):
    return max(1, (obj.max()-obj.min()).total_seconds())
def get_intervals_std(obj):
    if len(obj)<=2:
        return 1000
    intervals = []
    obj = obj.tolist()
    for i in range(len(obj)-1):
        intervals.append( (obj[i+1] - obj[i]).total_seconds() )
    std = np.array(intervals).std()
    return std

def generate_clicks_on_app_features(combine, features):
    for feature in features:
        derived_feature = feature+'_clicks_on_app_perc'
        start = time.time()
        total_clicks = combine.groupby(feature)[feature].transform('count')
        clicks_on_app = combine.groupby([feature, 'app'])[feature].transform('count')
        combine[derived_feature] = clicks_on_app/total_clicks
        print('{:.2f}s to generate feature {}'.format(time.time()-start, derived_feature))

        if feature=='ip':
            combine['ip_total_clicks'] = total_clicks.astype('uint16')
            print('feature ip_total_clicks created (no extra time)')
            combine['ip_total_clicks_on_app'] = clicks_on_app.astype('uint16')
            print('feature ip_total_clicks_on_app created (no extra time)')
        else:
            del clicks_on_app
            del total_clicks
        gc.collect()

def get_processed_data(data_train, data_test):
    start = time.time()
    combine = pd.concat([data_train, data_test])
    
    del data_train, data_test
    gc.collect()
    
    generate_clicks_on_app_features(combine, ['ip', 'os', 'device', 'channel'])
    
    start = time.time()
    combine['ip_os_total_clicks_on_app'] = combine.groupby(['ip', 'os', 'app'])['ip'].transform('count').astype('uint16')
    print('{:.2f}s to generate feature ip_os_total_clicks_on_app'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_os_total_clicks'] = combine.groupby(['ip', 'os'])['ip'].transform('count').astype('uint16')
    print('{:.2f}s to generate feature ip_os_total_clicks'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_os_clicks_on_app_perc'] = combine['ip_os_total_clicks_on_app']/combine['ip_os_total_clicks']
    print('{:.2f}s to generate feature ip_os_clicks_on_app_perc'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_activity_on_app_duration'] = combine.groupby(['ip', 'app'])['click_time'].transform(get_timediff).astype('uint32')
    print('{:.2f}s to generate feature ip_activity_on_app_duration'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_clicks_on_app_frequency'] = combine['ip_total_clicks_on_app']/combine['ip_activity_on_app_duration']
    print('{:.2f}s to generate feature ip_clicks_on_app_frequency'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_activity_intervals_std'] = combine.groupby('ip')['click_time'].transform(get_intervals_std)
    print('{:.2f}s to generate feature ip_activity_intervals_std'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_total_activity_duration'] = combine.groupby('ip')['click_time'].transform(get_timediff).astype('uint32')
    print('{:.2f}s to generate feature ip_total_activity_duration'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_clicks_frequency'] = combine['ip_total_clicks']/combine['ip_total_activity_duration']
    print('{:.2f}s to generate feature ip_total_activity_duration'.format(time.time()-start))
    gc.collect()
    
    start = time.time()
    combine['ip_activity_on_app_duration_perc'] = combine['ip_activity_on_app_duration']/combine['ip_total_activity_duration']
    print('{:.2f}s to generate feature ip_activity_on_app_duration_perc'.format(time.time()-start))
    gc.collect()
    
    combine.drop(columns=['ip', 'click_time'], inplace=True)
    gc.collect()
    
    replace_base_features_by_their_perc(combine, ['app', 'os', 'device', 'channel'])
    gc.collect()
    
    data_train = combine[:train_size].drop(columns=['click_id'])
    data_test = combine[train_size:].drop(columns=['is_attributed'])
    
    del combine
    gc.collect()
    
    return (data_train, data_test)


start = time.time()
data_train, data_test = get_processed_data(data_train, data_test)
data_test['click_id'] = data_test['click_id'].astype('uint32')
process_time = time.time()-start
print('{:.2f}s to process data ({:.2f} lines/s)'.format(process_time, (data_train.shape[0]+data_test.shape[0])/process_time))


# saving csv

start = time.time()
data_train.to_csv('input/train_processed.csv', index=False)
process_time = time.time()-start
print('{:.2f}s to write to csv ({:.2f} lines/s)'.format(process_time, data_train.shape[0]/process_time))

start = time.time()
data_test.to_csv('input/test_processed.csv', index=False)
process_time = time.time()-start
print('{:.2f}s to write to csv ({:.2f} lines/s)'.format(process_time, data_test.shape[0]/process_time))

print('{:.2f}s to run script'.format(time.time()-raw_start))