import argparse
import datetime
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 1212


def load_transactions(file_path: str) -> pd.DataFrame:
    dtypes = {
        'tran_id': int,
        'orig_acct': int,
        'bene_acct': int,
        'tx_type': str,
        'base_amt': float,
        'tran_timestamp': str,
        'is_sar': int,
        'alert_id': int
    }

    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['tran_timestamp'])

    return df


def generate_feature_vectors(input_file_path: str, output_filename: str):
    feat_df = load_transactions(input_file_path)
    feat_df.drop(['tran_id', 'alert_id'], axis=1, inplace=True)

    categories = pd.get_dummies(feat_df['tx_type'], dtype=int)

    for col in reversed(categories.columns):
        feat_df.insert(loc=2, column=str.lower(col), value=categories.loc[:, col])
    feat_df.drop('tx_type', axis=1, inplace=True)

    feat_df['tran_timestamp'] = (feat_df['tran_timestamp'] -
                                 pd.Timestamp('1970-01-01 00:00:00+00:00')) // pd.Timedelta('1s')

    train_x, dev_test_x, train_y, dev_test_y = train_test_split(feat_df.drop('is_sar', axis=1),
                                                                feat_df['is_sar'],
                                                                train_size=0.9,
                                                                shuffle=True,
                                                                stratify=feat_df['is_sar'],
                                                                random_state=SEED)

    dev_x, test_x, dev_y, test_y = train_test_split(dev_test_x,
                                                    dev_test_y,
                                                    train_size=0.5,
                                                    shuffle=True,
                                                    stratify=dev_test_y,
                                                    random_state=SEED)

    train_x.to_csv('data/features_train_%s.csv' % output_filename, index=False)
    dev_x.to_csv('data/features_dev_%s.csv' % output_filename, index=False)
    test_x.to_csv('data/features_test_%s.csv' % output_filename, index=False)

    train_y.to_csv('data/labels_train_%s.csv' % output_filename, index=False)
    dev_y.to_csv('data/labels_dev_%s.csv' % output_filename, index=False)
    test_y.to_csv('data/labels_test_%s.csv' % output_filename, index=False)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='AMLSim transactions data input file path.')
    parser.add_argument('output_filename', type=str, help='Output dataset filename.')

    args = parser.parse_args()

    return args


def generate_amlsim_data():
    args = arg_parser()
    generate_feature_vectors(args.input, args.output_filename)


if __name__ == '__main__':
    generate_amlsim_data()
