import numpy as np
import time
import gc

def get_attributed_count(obj):
    return np.nan_to_num(obj).sum()

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

def generate_attributed_perc_features(df, groupbys):
    for groupby in groupbys:
        prefix = '_'.join(groupby)+'_'

        start = time.time()
        attributed_count = df.groupby(groupby)['is_attributed'].transform(get_attributed_count)
        df[prefix+'attributed_perc'] = attributed_count/df[prefix+'clicks']
        del attributed_count
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'attributed_perc'))

def generate_time_related_features(df, groupbys):
    for groupby in groupbys:
        prefix = '_'.join(groupby)+'_'

        start = time.time()
        df[prefix+'duration'] = df.groupby(groupby)['click_time'].transform(get_timediff).astype('uint32')
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'duration'))

        start = time.time()
        df[prefix+'clicks_intervals_std'] = df.groupby(groupby)['click_time'].transform(get_intervals_std)
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_intervals_std'))

        start = time.time()
        df[prefix+'clicks_frequency'] = df[prefix+'clicks']/df[prefix+'duration']
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_frequency'))

        if prefix != 'app_':
            start = time.time()
            df[prefix+'duration_on_app'] = df.groupby(groupby+['app'])['click_time'].transform(get_timediff).astype('uint32')
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'duration_on_app'))

            start = time.time()
            df[prefix+'clicks_on_app_intervals_std'] = df.groupby(groupby+['app'])['click_time'].transform(get_intervals_std)
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_on_app_intervals_std'))

            start = time.time()
            df[prefix+'duration_on_app_perc'] = df[prefix+'duration_on_app']/df[prefix+'duration']
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'duration_on_app_perc'))

            start = time.time()
            df[prefix+'clicks_on_app_frequency'] = df[prefix+'clicks_on_app']/df[prefix+'duration_on_app']
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_on_app_frequency'))

def generate_count_related_features(df, groupbys):
    for groupby in groupbys:
        prefix = '_'.join(groupby)+'_'

        start = time.time()
        df[prefix+'clicks'] = df.groupby(groupby)['ip'].transform('count').astype('uint32')
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks'))

        if prefix != 'app_':
            start = time.time()
            df[prefix+'clicks_on_app'] = df.groupby(groupby+['app'])['ip'].transform('count').astype('uint32')
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_on_app'))

            start = time.time()
            df[prefix+'clicks_on_app_perc'] = df[prefix+'clicks_on_app']/df[prefix+'clicks']
            gc.collect()
            print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks_on_app_perc'))

def transform(df):
    groupbys = [['ip'], ['app'], ['os'], ['device'], ['channel'], ['ip', 'os'], ['ip', 'os', 'device'], ['ip', 'os', 'device', 'channel']]
    generate_count_related_features(df, groupbys)
    generate_time_related_features(df, groupbys)

    df.drop(columns=['ip'], inplace=True)

    groupbys = [['app'], ['os'], ['device'], ['channel']]
    generate_attributed_perc_features(df, groupbys)

    df['click_time'] = df['click_time'].apply(lambda x: x.hour).astype('uint8')

    df.drop(columns=['os', 'device', 'app', 'channel'], inplace=True)

    print('new dataframe shape: {}'.format(df.shape))
