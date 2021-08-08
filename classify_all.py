from classification import Classification
import pandas as pd
import pickle5 as pickle
import tf_idf_features as tf_feat
import sys
import json
import os

# all features including tfidf
# features subgroups


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

    train_apps = 'features/features_apps_train.csv'
    test_apps = 'features/features_apps_test.csv'

    train_filmes = 'features/features_movies_train.csv'
    test_filmes = 'features/features_movies_test.csv'

    # Tokens
    tk_tr_a = 'preprocessed/apps_train_tokens.pkl.gzip'
    tk_te_a = 'preprocessed/apps_test_tokens.pkl.gzip'
    tk_dev_a = 'preprocessed/apps_dev_tokens.pkl.gzip'

    tk_tr_f = 'preprocessed/movies_train_tokens.pkl.gzip'
    tk_te_f = 'preprocessed/movies_test_tokens.pkl.gzip'
    tk_dev_f = 'preprocessed/movies_dev_tokens.pkl.gzip'

    df_train = None
    df_test = None

    if domain == 'apps':
        df_train = pd.read_csv(train_apps, index_col=0)
        df_test = pd.read_csv(test_apps, index_col=0)
        columns = df_train.columns
        if sel_features == 'ig':
            features_ig_apps.append('helpfulness')
            df_train = df_train[features_ig_apps]
            df_test = df_test[features_ig_apps]
        elif sel_features == 'corr':
            features_corr_apps.append('helpfulness')
            df_train = df_train[features_corr_apps]
            df_test = df_test[features_corr_apps]
        elif sel_features == 'rf':
            features_rf_apps.append('helpfulness')
            df_train = df_train[features_rf_apps]
            df_test = df_test[features_rf_apps]

        if normalize:
            train_normal = pd.DataFrame(
                Classification.normalize(df_train.iloc[:, 0:-1]), columns=df_train.columns[:-1])
            test_normal = pd.DataFrame(
                Classification.normalize(df_test.iloc[:, 0:-1]), columns=df_train.columns[:-1])
            df_train = pd.concat(
                [train_normal, df_train['helpfulness']], axis=1)
            df_test = pd.concat(
                [test_normal, df_test['helpfulness']], axis=1)

        if tf or tfidf:
            tks_train = load_tokens(tk_tr_a)
            tks_test = load_tokens(tk_te_a)
            df_train = pd.concat([tks_train, df_train], axis=1)
            df_test = pd.concat([tks_test, df_test], axis=1)

    elif domain == 'movies':
        df_train = pd.read_csv(train_filmes, index_col=0)
        df_test = pd.read_csv(test_filmes, index_col=0)
        columns = df_train.columns
        if sel_features == 'ig':
            features_ig_movies.append('helpfulness')
            df_train = df_train[features_ig_movies]
            df_test = df_test[features_ig_movies]
        elif sel_features == 'corr':
            features_corr_movies.append('helpfulness')
            df_train = df_train[features_corr_movies]
            df_test = df_test[features_corr_movies]
        elif sel_features == 'rf':
            features_rf_movies.append('helpfulness')
            df_train = df_train[features_rf_movies]
            df_test = df_test[features_rf_movies]

        if normalize:
            train_normal = pd.DataFrame(
                Classification.normalize(df_train.iloc[:, 0:-1]), columns=df_train.columns[:-1])
            test_normal = pd.DataFrame(
                Classification.normalize(df_test.iloc[:, 0:-1]), columns=df_train.columns[:-1])
            df_train = pd.concat(
                [train_normal, df_train['helpfulness']], axis=1)
            df_test = pd.concat(
                [test_normal, df_test['helpfulness']], axis=1)

        if tf or tfidf:
            tks_train = load_tokens(tk_tr_f)
            tks_test = load_tokens(tk_te_f)
            df_train = pd.concat([tks_train, df_train], axis=1)
            df_test = pd.concat([tks_test, df_test], axis=1)
    else:
        print("Verifique domínio, saindo")
        exit()

    count_class_0, count_class_1 = df_train.helpfulness.value_counts()
    df_class_0 = df_train[df_train['helpfulness'] == 0]
    df_class_1 = df_train[df_train['helpfulness'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)

    df_train = pd.concat([df_class_0_under, df_class_1], axis=0)

    # df_train_under = pd.read_pickle('train_t.pkl')

    count_class_0, count_class_1 = df_test.helpfulness.value_counts()
    df_class_0 = df_test[df_test['helpfulness'] == 0]
    df_class_1 = df_test[df_test['helpfulness'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)

    df_test = pd.concat([df_class_0_under, df_class_1], axis=0)

    len_train = df_train.shape[0]
    len_test = df_test.shape[0]

    if tf:
        df = pd.concat([df_train, df_test])
        X, features_names = tf_feat.get_bow(df['text'], 500)
        dense_train = pd.DataFrame(
            X.todense()[:len_train], columns=features_names, index=df_train.index)
        dense_test = pd.DataFrame(
            X.todense()[len_train:], columns=features_names, index=df_test.index)

        df_train = pd.concat(
            [dense_train, df_train.iloc[:, 1:]], axis=1)

        df_test = pd.concat(
            [dense_test, df_test.iloc[:, 1:]], axis=1)
    elif tfidf:
        df = pd.concat([df_train, df_test])

        X, features_names = tf_feat.get_tfidf_sklearn(
            df['text'], 500)

        dense_train = pd.DataFrame(
            X.todense()[:len_train], columns=features_names, index=df_train.index)
        dense_test = pd.DataFrame(
            X.todense()[len_train:], columns=features_names, index=df_test.index)

        df_train = pd.concat(
            [dense_train, df_train.iloc[:, 1:]], axis=1)

        df_test = pd.concat(
            [dense_test, df_test.iloc[:, 1:]], axis=1)
    else:
        pass

    return df_train.fillna(0), df_test.fillna(0)


if __name__ == '__main__':
    domain = sys.argv[1]
    method = sys.argv[2]
    model_folder = sys.argv[3]
    result_folder = sys.argv[4]
    tf = True if sys.argv[6] == '1' else False
    tfidf = True if sys.argv[7] == '1' else False

    tf_idf = 'tf_' if tf else ''
    tf_idf += 'tfidf' if tfidf else ''

    sel_features = 'all'
    try:
        sel_features = sys.argv[5]
    except IndexError:
        sel_features = 'all'

    metodos = ['naive_bayes', 'svm', 'tree', 'nn', 'randfor', 'oner', 'dummy']

    if method not in metodos:
        print("Método inválido... Saindo")
        exit()

    name_model = "%s_%s_%s_%s.model" % (domain, method, sel_features, tf_idf)
    name_result = "%s_%s_%s_%s.json" % (domain, method, sel_features, tf_idf)

    df_train, df_test = get_features(
        domain, tf=tf, tfidf=tfidf, sel_features=sel_features)

    # print(df_train)
    # print(df_test)

    classificador = Classification(
        method, df_train.iloc[:, 0:-1], df_train['helpfulness'], df_test.iloc[:, 0:-1], df_test['helpfulness'], df_train.columns)

    resultado = classificador.classifier(
        os.sep.join([model_folder, name_model]))

    with open(os.sep.join([result_folder, name_result]), 'w') as f:
        json.dump(resultado, f)
