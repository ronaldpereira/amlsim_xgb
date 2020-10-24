import argparse
import pickle
from time import time
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def read_data(dataset_name: str):
    train_x = pd.read_csv('data/features_train_%s.csv' % dataset_name)
    dev_x = pd.read_csv('data/features_dev_%s.csv' % dataset_name)
    test_x = pd.read_csv('data/features_test_%s.csv' % dataset_name)

    train_y = pd.read_csv('data/labels_train_%s.csv' % dataset_name)
    dev_y = pd.read_csv('data/labels_dev_%s.csv' % dataset_name)
    test_y = pd.read_csv('data/labels_test_%s.csv' % dataset_name)

    return train_x, dev_x, test_x, train_y, dev_y, test_y


def search_hyperparams(train_x: pd.DataFrame, train_y: pd.Series, dev_x: pd.DataFrame,
                       dev_y: pd.Series, dataset_name: str):
    """Searches for the best hyperparameters for a given dataset for XGBoost using GridSearch with
    CrossValidation in order to maximize F1 Score metric.

    Args:
        train_x (pd.DataFrame): [description]
        train_y (pd.Series): [description]
        dev_x (pd.DataFrame): [description]
        dev_y (pd.Series): [description]
        dataset_name (str): [description]
    """

    params = {'objective': 'binary:logistic', 'n_estimators': 100, 'n_jobs': -1}

    xgb_model = xgb.XGBClassifier(**params)

    params_search = {
        'learning_rate': [0.01, 0.1, 0.5, 1],
        'max_depth': [5, 10, 15, 20],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.1, 0.5, 1]
    }

    f1_score_custom_scoring = make_scorer(f1_score, greater_is_better=True)

    clf = GridSearchCV(xgb_model, params_search, scoring=f1_score_custom_scoring, cv=3, n_jobs=-1)

    clf.fit(train_x, train_y)

    results_df = pd.DataFrame(clf.cv_results_)

    results_df = results_df.sort_values(by='rank_test_score')

    results_df.to_csv('output/xgb_results_grid_search_%s.csv' % dataset_name, index=False)

    with open('model/best_xgb_model_%s.pickle' % dataset_name, 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

    train_model(train_x, train_y, dev_x, dev_y, dataset_name, xgb_model=True)


def f1_score_custom(y_pred: np.array, y_true: xgb.DMatrix) -> Tuple[str, float]:
    """Custom F1 Score function for XGBoost custom evaluation metric definition.
    This F1 Score works for the binary average, using the fraudulent class as positive class.
    In this way, we can optmize the model to detect more and better fraudulent transactions.

    Args:
        y_pred (np.array): Prediction array.
        y_true (xgb.DMatrix): Ground-truth DMatrix that we'll use the get its labels from.

    Returns:
        Tuple[str, float]: Tuple containing the name of the metric and its value.
    """

    y_true = y_true.get_label()
    y_pred = y_pred > 0.5

    return 'f1_err', 1 - f1_score(y_true, y_pred, average='binary')


def train_model(train_x: pd.DataFrame,
                train_y: pd.Series,
                dev_x: pd.DataFrame,
                dev_y: pd.Series,
                dataset_name: str,
                xgb_model: bool = False):
    """Function that trains and evaluates the model, as well as save the pickle fitted model.

    Args:
        train_x (pd.DataFrame): Training features DataFrame.
        train_y (pd.Series): Training labels.
        dev_x (pd.DataFrame): Dev features DataFrame.
        dev_y (pd.Series): Dev labels.
        dataset_name (str): AMLSim dataset name.
        xgb_model (bool, optional): Whether to load the pickled model or not. Defaults to False.
    """

    if not xgb_model:
        params = {'objective': 'binary:logistic', 'n_estimators': 200, 'n_jobs': -1}
        params.update({
            'gamma': 0,
            'learning_rate': 0.5,
            'max_depth': 10,
            'reg_alpha': 0.5,
            'reg_lambda': 0
        })
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(train_x,
                      train_y,
                      eval_set=[(train_x, train_y), (dev_x, dev_y)],
                      eval_metric=f1_score_custom)

    else:
        with open('model/best_xgb_model_%s.pickle' % dataset_name, 'rb') as f:
            xgb_model = pickle.load(f)

    start = time()
    y_pred = xgb_model.predict(test_x)

    y_pred = y_pred > 0.5

    print('time spent for prediction: %f' % (time() - start))
    print('f1_score_binary for non frauds on test set: %f' %
          f1_score(test_y, y_pred, average='binary', pos_label=0))
    print('f1_score_binary for frauds on test set: %f' %
          f1_score(test_y, y_pred, average='binary', pos_label=1))
    print('f1_score_macro on test set: %f' % f1_score(test_y, y_pred, average='macro'))

    with open('model/xgb_model_%s.pickle' % dataset_name, 'wb') as f:
        pickle.dump(xgb_model, f)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')

    parser.add_argument('-s',
                        '--search',
                        action='store_true',
                        default=False,
                        help='Enable GridSearchCV mode')

    parser.add_argument('-l',
                        '--load_best_model',
                        action='store_true',
                        default=False,
                        help='Loads best_xgb_model_cv.pickle from model/ folder')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    train_x, dev_x, test_x, train_y, dev_y, test_y = read_data(args.dataset)
    if args.search:
        search_hyperparams(train_x, train_y, dev_x, dev_y, args.dataset)
    else:
        train_model(train_x, train_y, dev_x, dev_y, args.dataset, xgb_model=args.load_best_model)
