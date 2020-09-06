import argparse
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def read_data(dataset):
    train_x = pd.read_csv('data/features_train_%s.csv' % dataset)
    dev_x = pd.read_csv('data/features_dev_%s.csv' % dataset)
    test_x = pd.read_csv('data/features_test_%s.csv' % dataset)

    train_y = pd.read_csv('data/labels_train_%s.csv' % dataset).values.ravel()
    dev_y = pd.read_csv('data/labels_dev_%s.csv' % dataset).values.ravel()
    test_y = pd.read_csv('data/labels_test_%s.csv' % dataset).values.ravel()

    return train_x, dev_x, test_x, train_y, dev_y, test_y


def search_hyperparams(train_x, train_y, dev_x, dev_y, gpu=False):
    params = {'objective': 'binary:logistic', 'n_estimators': 100, 'n_jobs': -1}
    if gpu:
        params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})

    xgb_model = xgb.XGBClassifier(**params)

    params_search = {
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1, 5],
        'gamma': [0, 0.1, 0.5, 1, 10],
        'reg_alpha': [0, 0.1, 0.5, 1, 10],
        'reg_lambda': [0, 0.1, 0.5, 1, 10]
    }

    f1_score_custom_scoring = make_scorer(f1_score, greater_is_better=True)

    clf = GridSearchCV(xgb_model, params_search, scoring=f1_score_custom_scoring, cv=5, n_jobs=-1)

    clf.fit(train_x, train_y)

    results_df = pd.DataFrame(clf.cv_results_)

    results_df.to_csv('data/results_grid_search_cv.csv', index=False)

    best_model = clf.best_estimator_

    with open('model/best_xgb_model_cv.pickle', 'wb') as f:
        pickle.dump(best_model, f)

    train_model(train_x, train_y, dev_x, dev_y, xgb_model=True)


def f1_score_custom(y_pred, y_true):
    y_true = y_true.get_label()
    y_pred = y_pred > 0.5

    return 'f1_err', 1 - f1_score(y_true, y_pred, average='binary')


def train_model(train_x, train_y, dev_x, dev_y, gpu=False, xgb_model=False):
    params = {'objective': 'binary:logistic', 'n_estimators': 300, 'n_jobs': -1}

    if gpu:
        params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})

    if not xgb_model:
        params.update({'gamma': 0, 'learning_rate': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 0})
        xgb_model = xgb.XGBClassifier(**params)

    else:
        with open('model/best_xgb_model_cv.pickle', 'rb') as f:
            xgb_model = pickle.load(f)

    xgb_model.fit(train_x,
                  train_y,
                  eval_set=[(train_x, train_y), (dev_x, dev_y)],
                  eval_metric=f1_score_custom)

    y_pred = xgb_model.predict(test_x)

    y_pred = y_pred > 0.5

    print('f1_score_binary for non frauds on test set: %f' %
          f1_score(test_y, y_pred, average='binary', pos_label=0))
    print('f1_score_binary for frauds on test set: %f' %
          f1_score(test_y, y_pred, average='binary', pos_label=1))
    print('f1_score_macro on test set: %f' % f1_score(test_y, y_pred, average='macro'))

    with open('model/xgb_model.pickle', 'wb') as f:
        pickle.dump(xgb_model, f)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')

    parser.add_argument('-s',
                        '--search',
                        action='store_true',
                        default=False,
                        help='Enable GridSearchCV mode')

    parser.add_argument('-g',
                        '--gpu',
                        action='store_true',
                        default=False,
                        help='Enable GPU XGBoost training')

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
        search_hyperparams(train_x, train_y, dev_x, dev_y, gpu=args.gpu)
    else:
        train_model(train_x, train_y, dev_x, dev_y, gpu=args.gpu, xgb_model=args.load_best_model)
