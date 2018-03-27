import pandas as pd
import numpy as np
import time
import gc

def sort_features_by_attr_proba(df, features):
    sorted_features = {}

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

        indexes = {}
        for feature_id in feature_ids:
            indexes[feature_id] = sorted_ids.index(feature_id)
        del feature_ids, sorted_ids
        gc.collect()

        df[feature] = df[feature].apply(lambda x: indexes[x]).astype('uint16')
        sorted_features[feature] = indexes

        print('{:.2f}s to sort {}'.format(time.time()-start, feature))

    return sorted_features

def generate_count_features(df, groupbys):
    for groupby in groupbys:
        print('--- grouping by {}'.format(groupby))
        suffix = '_'+('_'.join(groupby))

        start = time.time()
        df['n'+suffix] = df.groupby(groupby)['ip'].transform('count').astype('uint32')
        gc.collect()
        print('{:.2f}s to generate feature {}'.format(time.time()-start, 'n'+suffix))

def transform(df):
    start = time.time()
    df['hour'] = pd.to_datetime(df['click_time']).dt.hour.astype('uint8')
    print('{:.2f}s to generate feature hour'.format(time.time()-start))

    groupbys = [['ip'], ['ip', 'app'], ['ip', 'os'], ['ip', 'device'],
	            ['ip', 'app', 'hour'], ['ip', 'os', 'hour'], ['ip', 'device', 'hour']]
    generate_count_features(df, groupbys)

    sorted_features = sort_features_by_attr_proba(df, ['app', 'os', 'device', 'channel'])

    return sorted_features
