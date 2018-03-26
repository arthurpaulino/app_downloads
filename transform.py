import pandas as pd
import numpy as np
import time
import gc

def sort_features_by_attr_proba(df, features):
    for feature in features:
        print('--- sorting {}'.format(feature))
        start = time.time()

        def compute_score(feature_id):
            attrs = df[(df[feature]==feature_id) & (df['is_attributed']==1)]
            n_attrs = attrs.shape[0]
            del attrs
            gc.collect()

            clicks = df[df[feature]==feature_id]
            n_clicks = clicks.shape[0]
            del clicks
            gc.collect()

            if n_attrs==0:
                return -n_clicks
            return n_attrs/n_clicks

        scores = {}
        feature_ids = set(df[feature])
        for feature_id in feature_ids:
            scores[feature_id] = compute_score(feature_id)
        sorted_ids = sorted(feature_ids, key=lambda feature_id: scores[feature_id], reverse=True)
        del scores
        gc.collect()

        df[feature] = df[feature].apply(lambda x: sorted_ids.index(x)).astype('uint16')
        del sorted_ids
        gc.collect()

        print('{:.2f}s to sort {}'.format(time.time()-start, feature))

def generate_count_features(df, groupbys):
    for groupby in groupbys:
        print('--- grouping by {}'.format(groupby))
        prefix = '_'.join(groupby)+'_'

        start = time.time()
        df[prefix+'clicks'] = df.groupby(groupby)['ip'].transform('count').astype('uint32')
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, prefix+'clicks'))

def transform(df):

    sort_features_by_attr_proba(df, ['app', 'os', 'channel'])

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
