import numpy as np
from pprint import pprint
import sys
import pandas as pd
from classification import Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
import json
from classify_all import get_features as gf
import datetime


def grid_randfor(X_train, Y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    pprint(random_grid)

    clf = RandomForestClassifier()

    halving_cv = HalvingGridSearchCV(
        clf, random_grid, scoring="f1_macro", n_jobs=-1, min_resources="exhaust", factor=3)

    halving_cv.fit(X_train, Y_train)

    best_estimator = halving_cv.best_estimator_
    best_score = halving_cv.best_score_
    best_params = halving_cv.best_params_

    return (best_estimator, best_score, best_params)


def grid_svm(X_train, Y_train):
    param_grid = {
        'C': np.arange(0.01, 100, 10)
    }

    clf = LinearSVC()
    halving_cv = HalvingGridSearchCV(
        clf, param_grid, scoring="f1_macro", n_jobs=-1, min_resources="exhaust", factor=3)

    halving_cv.fit(X_train, Y_train)

    best_estimator = halving_cv.best_estimator_
    best_score = halving_cv.best_score_
    best_params = halving_cv.best_params_

    return (best_estimator, best_score, best_params)


def get_features(domain, normalize=True):
    train_apps = 'features/features_apps_train.csv'
    test_apps = 'features/features_apps_test.csv'

    train_filmes = 'features/features_movies_train.csv'
    test_filmes = 'features/features_movies_test.csv'

    features_df_train = None
    features_df_test = None

    if domain == 'apps':
        features_df_train = pd.read_csv(train_apps, index_col=0)
        features_df_test = pd.read_csv(test_apps, index_col=0)
        columns = features_df_train.columns
    elif domain == 'movies':
        features_df_train = pd.read_csv(train_filmes, index_col=0)
        features_df_test = pd.read_csv(test_filmes, index_col=0)
        columns = features_df_train.columns
    else:
        print("Verifique domínio, saindo")
        exit()

    count_class_0, count_class_1 = features_df_train.helpfulness.value_counts()
    df_class_0 = features_df_train[features_df_train['helpfulness'] == 0]
    df_class_1 = features_df_train[features_df_train['helpfulness'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)
    ##
    # df_class_0_under = df_class_0.sample(limit_test)
    # df_class_1_under = df_class_1.sample(limit_test)
    # df_train_under = pd.concat([df_class_0_under, df_class_1_under], axis=0)
    ##
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    # df_train_under = pd.read_pickle('train_t.pkl')

    count_class_0, count_class_1 = features_df_test.helpfulness.value_counts()
    df_class_0 = features_df_test[features_df_test['helpfulness'] == 0]
    df_class_1 = features_df_test[features_df_test['helpfulness'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)

    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    if normalize:
        X_train = Classification.normalize(df_train_under.iloc[:, 0:-1])
        X_test = Classification.normalize(df_test_under.iloc[:, 0:-1])
    else:
        X_train = df_train_under.iloc[:, 0:-1]
        X_test = df_test_under.iloc[:, 0:-1]

    Y_train = df_train_under['helpfulness']
    Y_test = df_test_under['helpfulness']

    return X_train, Y_train, X_test, Y_test


print(datetime.datetime.now())

domain = sys.argv[1]
model_folder = sys.argv[2]
method = sys.argv[3]
sel_features = sys.argv[4]
# result_folder = sys.argv[4]
# sel_features = sys.argv[5]

if sel_features == 'all':
    df_train, df_test = gf(domain)
    X_train = df_train.iloc[:, 0:-1].fillna(0)
    Y_train = df_train['helpfulness']
    X_test = df_test.iloc[:, 0:-1].fillna(0)
    Y_test = df_test['helpfulness']
elif sel_features == 'hand':
    X_train, Y_train, X_test, Y_test = get_features(domain)

if method == 'rf':
    clf, score, params = grid_randfor(
        X_train, Y_train)
elif method == 'svm':
    clf, score, params = grid_svm(
        X_train, Y_train)
else:
    print("Método inválido")
    exit()

y_pred = clf.predict(X_test)


resultado = classification_report(
    Y_test, y_pred, output_dict=True)

with open(os.sep.join([model_folder, '%s_grid_%s_%s.result' % (method, domain, sel_features)]), 'w') as f:
    json.dump(resultado, f)

with open(os.sep.join(
        [model_folder, '%s_model_grid_%s_%s.params' % (method, domain, sel_features)]), 'w') as f:
    f.write(str(score)+'\n')
    f.write(str(params)+'\n')


joblib.dump(clf, os.sep.join(
    [model_folder, '%s_model_grid_%s_%s.model' % (method, domain, sel_features)]))

print(datetime.datetime.now())
