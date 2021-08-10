from classification import Classification
import pandas as pd
import pickle5 as pickle
import tf_idf_features as tf_feat
from sklearn.metrics import classification_report
import joblib
import sys
import json
import os
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from classify_all import get_features as gf

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV


def load_tokens(df_tk_path):
    with open(df_tk_path, "rb") as fh:
        tks = pickle.load(fh)

    tks = tks[tks.tokens.str.len() > 0]

    tks = tks['tokens'].apply(
        lambda x: ' '.join([a.lower() for a, _, _ in x]))
    tks = tks.to_frame(name='text')
    return tks


def get_features(domain, tf=False, tfidf=True, sel_features='all', normalize=True):

    # Features
    features_ig_apps = ['stars_deviation', 'num_words',
                        'GF', 'avg_sent_len', 'S', 'CI', 'FKI', 'nouns']
    features_corr_apps = ['num_words', 'avg_sent_len',
                          'spell', 'swear', 'affect', 'aspects', 'negemo', 'ARI']
    features_rf_apps = ['stars_deviation', 'num_words',
                        'CI', 'avg_sent_len', 'ARI', 'S', 'GF', 'FI']

    features_ig_movies = ['stars_deviation', 'num_words',
                          'FI', 'GF', 'avg_sent_len', 'S', 'CI', 'nouns']
    features_corr_movies = ['num_words', 'swear', 'avg_sent_len',
                            'spell', 'negemo', 'SMOG', 'affect', 'num_sents']
    features_rf_movies = ['stars_deviation', 'CI',
                          'ARI', 'GF', 'FI', 'FKI', 'S', 'avg_sent_len']

    # train_apps = 'features/features_apps_train.csv'
    # test_apps = 'features/features_apps_test.csv'
    val_apps = 'features/features_apps_dev.csv'

    # train_filmes = 'features/features_movies_train.csv'
    # test_filmes = 'features/features_movies_test.csv'
    val_movies = 'features/features_movies_dev.csv'
    # Tokens
    # tk_tr_a = 'preprocessed/apps_train_tokens.pkl.gzip'
    # tk_te_a = 'preprocessed/apps_test_tokens.pkl.gzip'
    tk_dev_a = 'preprocessed/apps_dev_tokens.pkl.gzip'

    # tk_tr_f = 'preprocessed/movies_train_tokens.pkl.gzip'
    # tk_te_f = 'preprocessed/movies_test_tokens.pkl.gzip'
    tk_dev_f = 'preprocessed/movies_dev_tokens.pkl.gzip'

    # df_train = None
    # df_test = None
    df_val = None

    if domain == 'apps':
        # df_train = pd.read_csv(train_apps, index_col=0)
        # df_test = pd.read_csv(test_apps, index_col=0)
        df_val = pd.read_csv(val_apps, index_col=0)
        columns = df_val.columns

        if normalize:
            val_normal = pd.DataFrame(
                Classification.normalize(df_val.iloc[:, 0:-1]), columns=df_val.columns[:-1])
            df_train = pd.concat(
                [val_normal, df_val['helpfulness']], axis=1)

        if tf or tfidf:
            tks_val = load_tokens(tk_dev_a)

            df_val = pd.concat([tks_val, df_val], axis=1)

    elif domain == 'movies':
        df_val = pd.read_csv(val_movies, index_col=0)

        columns = df_val.columns

        if normalize:
            val_normal = pd.DataFrame(
                Classification.normalize(df_val.iloc[:, 0:-1]), columns=df_val.columns[:-1])

            df_val = pd.concat(
                [val_normal, df_val['helpfulness']], axis=1)

        if tf or tfidf:
            tks_val = load_tokens(tk_dev_f)

            df_val = pd.concat([tks_val, df_val], axis=1)

    else:
        print("Verifique domínio, saindo")
        exit()

    count_class_0, count_class_1 = df_val.helpfulness.value_counts()

    df_class_0 = df_val[df_val['helpfulness'] == 0]
    df_class_1 = df_val[df_val['helpfulness'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)

    df_val = pd.concat([df_class_0_under, df_class_1], axis=0)

    # df_train_under = pd.read_pickle('train_t.pkl')

    # len_val = df_val.shape[0]

    if tf:
        #df = pd.concat([df_train, df_test])
        X, features_names = tf_feat.get_bow(df_val['text'], 500)
        dense_val = pd.DataFrame(
            X.todense(), columns=features_names, index=df_val.index)

        df_val = pd.concat(
            [dense_val, df_val.iloc[:, 1:]], axis=1)

    elif tfidf:
        #df = pd.concat([df_train, df_test])

        X, features_names = tf_feat.get_tfidf_sklearn(
            df_val['text'], 500)

        dense_val = pd.DataFrame(
            X.todense(), columns=features_names, index=df_val.index)

        df_val = pd.concat(
            [dense_val, df_val.iloc[:, 1:]], axis=1)
    else:
        pass

    return df_val.fillna(0)


results_folder = "/media/rogerio/Novo volume/MODELS_ALL_TFIDF/"

nb_apps = joblib.load(os.sep.join(
    [results_folder, 'apps_naive_bayes_all_tfidf.model.model']))

svm_apps = joblib.load(os.sep.join(
    [results_folder, 'apps_svm_all_tfidf.model.model']))

dt_apps = joblib.load(os.sep.join(
    [results_folder, 'apps_tree_all_tfidf.model.model']))

nn_apps = joblib.load(os.sep.join(
    [results_folder, 'apps_nn_all_tfidf.model.model']))

rf_apps = joblib.load(os.sep.join(
    [results_folder, 'apps_randfor_all_tfidf.model.model']))

apps_models = {'NB': nb_apps, 'SVM': svm_apps,
               'DT': dt_apps, 'nn': nn_apps, 'rf': rf_apps}


nb_movies = joblib.load(os.sep.join(
    [results_folder, 'movies_naive_bayes_all_tfidf.model.model']))

svm_movies = joblib.load(os.sep.join(
    [results_folder, 'movies_svm_all_tfidf.model.model']))

dt_movies = joblib.load(os.sep.join(
    [results_folder, 'movies_tree_all_tfidf.model.model']))

nn_movies = joblib.load(os.sep.join(
    [results_folder, 'movies_nn_all_tfidf.model.model']))

rf_movies = joblib.load(os.sep.join(
    [results_folder, 'movies_randfor_all_tfidf.model.model']))

movies_models = {'NB': nb_movies, 'SVM': svm_movies,
                 'DT': dt_movies, 'nn': nn_movies, 'rf': rf_movies}

#ds_apps_train, ds_apps_test = gf('apps')
ds_movies_train, ds_movies_test = gf('movies')
#ds_apps = get_features('apps')
#ds_movies = get_features('movies')

# print("APPS")
# for name_1, model_1 in apps_models.items():
#     for name_2, model_2 in apps_models.items():
#         if name_1 != name_2:
#             y_pred_1 = model_1.predict(ds_apps.iloc[:, 0:-1])
#             y_pred_2 = model_2.predict(ds_apps.iloc[:, 0:-1])

#             erros_1 = y_pred_1 != ds_apps['helpfulness']
#             erros_2 = y_pred_2 != ds_apps['helpfulness']

#             erros_comum = erros_1 & erros_2
#             intersection = np.count_nonzero(erros_comum)
#             print("%s x %s: %s" % (name_1, name_2, intersection))


# print("MOVIES")
# for name_1, model_1 in movies_models.items():
#     for name_2, model_2 in movies_models.items():
#         if name_1 != name_2:
#             y_pred_1 = model_1.predict(ds_movies.iloc[:, 0:-1])
#             y_pred_2 = model_2.predict(ds_movies.iloc[:, 0:-1])

#             erros_1 = y_pred_1 != ds_movies['helpfulness']
#             erros_2 = y_pred_2 != ds_movies['helpfulness']

#             erros_comum = erros_1 & erros_2
#             intersection = np.count_nonzero(erros_comum)
#             print("%s x %s: %s" % (name_1, name_2, intersection))
# y_pred = loaded_model.predict(dataset.iloc[:, 0:-1])

clf1 = LinearSVC(random_state=0, tol=1e-05)
clf2 = DecisionTreeClassifier()
clf3 = MLPClassifier(solver='adam', hidden_layer_sizes=(
    20, 20), random_state=42, max_iter=1000)
criterion = 'entropy'  # best fit
estimators = 200  # best fit
max_depth = None  # best fit
clf4 = RandomForestClassifier(
    criterion=criterion, n_estimators=estimators, max_depth=max_depth, n_jobs=-1)

eclf1 = VotingClassifier(estimators=[('dt', clf2), (
    'rf', clf4)], voting='hard')

# eclf1 = VotingClassifier(estimators=[('dt', movies_models['DT']), (
# 'rf', movies_models['rf'])], voting='hard', n_jobs=-1)

# X_train, X_test, y_train, y_test = train_test_split(
#     ds_apps.iloc[:, 0:-1], ds_apps['helpfulness'], test_size=0.33, random_state=42)

eclf1.fit(ds_movies_train.iloc[:, 0:-1], ds_movies_train['helpfulness'])

y_pred = eclf1.predict(ds_movies_test.iloc[:, 0:-1])

print('svm - dt')
print(classification_report(y_pred, ds_movies_test['helpfulness']))
