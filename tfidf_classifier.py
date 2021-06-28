from os import name
from read_corpus import load_corpus
import pandas as pd
from classification import Classification
import tf_idf_features as tf_feat
from sklearn.model_selection import train_test_split
import pprint
import sys
import json

domain = sys.argv[1]
method = sys.argv[2]
# model_folder = sys.argv[3]
result_folder = sys.argv[3]
sel_features = sys.argv[4]


def load_dataset_local(df_tk_path, df_help_path):
    feat_apps_dev = pd.read_csv(df_help_path, index_col=0)
    tk_apps_dev = pd.read_pickle(df_tk_path)

    tk_apps_dev = tk_apps_dev[tk_apps_dev.tokens.str.len() > 0]

    tk_apps_dev = tk_apps_dev['tokens'].apply(
        lambda x: ' '.join([a.lower() for a, _, _ in x]))
    tk_apps_dev = tk_apps_dev.to_frame(name='text')

    df = pd.concat([tk_apps_dev, feat_apps_dev['helpfulness']], axis=1)

    return df


    # dominios = ['features/features_apps_dev.csv', 'features/features_movies_dev']
feat_tr_a = 'features/features_apps_train.csv'
feat_te_a = 'features/features_apps_test.csv'
feat_dev_a = 'features/features_apps_dev.csv'

feat_tr_f = 'features/features_movies_train.csv'
feat_te_f = 'features/features_movies_test.csv'
feat_dev_f = 'features/features_movies_dev.csv'

tk_tr_a = 'preprocessed/apps_train_tokens.pkl.gzip'
tk_te_a = 'preprocessed/apps_test_tokens.pkl.gzip'
tk_dev_a = 'preprocessed/apps_dev_tokens.pkl.gzip'

tk_tr_f = 'preprocessed/movies_train_tokens.pkl.gzip'
tk_te_f = 'preprocessed/movies_test_tokens.pkl.gzip'
tk_dev_f = 'preprocessed/movies_dev_tokens.pkl.gzip'


# dominios = {'apps_dev': file_path_apps_dev, 'movies_dev': file_path_movies_dev, 'apps_test': file_path_apps_test,
#            'movies_test': file_path_movies_test, 'apps_train': file_path_apps_train, 'movies_train': file_path_movies_train}

df_train = None
df_test = None

if domain == 'apps':
    df_train = load_dataset_local(tk_tr_a, feat_tr_a)
    df_test = load_dataset_local(tk_te_a, feat_te_a)
elif domain == 'movies':
    df_train = load_dataset_local(tk_tr_f, feat_tr_f)
    df_test = load_dataset_local(tk_te_f, feat_te_f)
else:
    print("Verifique domínio, saindo")
    exit()

metodos = ['naive_bayes', 'tree', 'randfor', 'oner', 'svm', 'nn']

if method not in metodos:
    print("Método inválido... Saindo")
    exit()

# print(tf_feat.get_bow(df['text']))
# print(tf_feat.get_tfidf_sklearn(df['text']))

count_class_0, count_class_1 = df_train.helpfulness.value_counts()
df_class_0 = df_train[df_train['helpfulness'] == 0]
df_class_1 = df_train[df_train['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
##
# df_class_0_under = df_class_0.sample(limit_test)
# df_class_1_under = df_class_1.sample(limit_test)
# df_train_under = pd.concat([df_class_0_under, df_class_1_under], axis=0)
##

count_class_0, count_class_1 = df_test.helpfulness.value_counts()
df_class_0 = df_test[df_test['helpfulness'] == 0]
df_class_1 = df_test[df_test['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

len_train = df_train_under.shape[0]
len_test = df_test_under.shape[0]

df = pd.concat([df_train_under, df_test_under])

if sel_features == 'tf':
    X, features_names = tf_feat.get_bow(df['text'], 1000)
elif sel_features == 'tfidf':
    X, features_names = tf_feat.get_tfidf_sklearn(df['text'], 1000)
else:
    sel_features = 'tf'
    X, features_names = tf_feat.get_bow(df['text'], 1000)

# name_model = "%s_%s_%s.model" % (domain, method, sel_features)
name_result = "%s_%s_%s.json" % (domain, method, sel_features)

# X_train, X_test, y_train, y_test = train_test_split(
#    X.todense(), df_train_under['helpfulness'], test_size=0.2, random_state=42)

classificador = Classification(
    method, X.todense()[:len_train], df_train_under['helpfulness'], X.todense()[len_train:], df_test_under['helpfulness'], features_names)

resultado = classificador.classifier()
with open(result_folder+name_result, 'w') as f:
    json.dump(resultado, f)
