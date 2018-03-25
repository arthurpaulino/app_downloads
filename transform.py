import pandas as pd
import numpy as np
import time
import gc

def generate_count_features(df, groupbys):
    for groupby in groupbys:
        print('--- grouping by {}'.format(groupby))
        prefix = '_'.join(groupby)+'_'

        start = time.time()
        df[prefix+'clicks'] = df.groupby(groupby)['ip'].transform('count').astype('uint32')
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks'))

def transform(df):
    start = time.time()
    df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
    df.drop(columns=['click_time'], inplace=True)
    gc.collect()
    print('{:.2f}s to generate feature hour'.format(time.time()-start))

    groupbys = [['ip', 'os', 'hour'], ['ip', 'hour']]
    generate_count_features(df, groupbys)

    groupbys = [['ip'],        ['ip', 'os'],        ['ip', 'channel'],
                ['ip', 'app'], ['ip', 'os', 'app'], ['ip', 'channel', 'app']]
    generate_count_features(df, groupbys)

    df.drop(columns=['ip'], inplace=True)
    gc.collect()

    print('new dataframe shape: {}'.format(df.shape))
