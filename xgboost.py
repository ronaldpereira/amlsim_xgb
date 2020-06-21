import xgboost as xgb
import pandas as pd
import argparse


def read_data():
    train_x = pd.read_csv('data/features_train_amlsim.csv')
    dev_x = pd.read_csv('data/features_dev_amlsim.csv')
    test_x = pd.read_csv('data/features_test_amlsim.csv')

    train_y = pd.read_csv('data/labels_train_amlsim.csv')
    dev_y = pd.read_csv('data/labels_dev_amlsim.csv')
    test_y = pd.read_csv('data/labels_test_amlsim.csv')

    return train_x, dev_x, test_x, train_y, dev_y, test_y


def train_model(train_x, train_y, dev_x, dev_y):
    xgb_model = xgb.XGBClassifier(max_depth=15, n_jobs=-1)

    xgb_model.fit(train_x, train_y, eval_set=[(train_x, train_y), (dev_x, dev_y)])


if __name__ == "__main__":
    train_x, dev_x, test_x, train_y, dev_y, test_y = read_data()
    xgb_model = train_model(train_x, train_y, dev_x, dev_y)
