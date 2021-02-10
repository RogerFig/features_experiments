import pandas as pd
from classification import Classification
import pprint
import sys
import json

print(sys.argv)

domain = sys.argv[1]
method = sys.argv[2]
model_folder = sys.argv[3]
result_folder = sys.argv[4]


dominios = ['features/features_apps_dev.csv', 'features/features_movies_dev']
train_apps = 'features/features_apps_train.csv'
test_apps = 'features/features_apps_test.csv'

train_filmes = 'features/features_movies_train.csv'
test_filmes = 'features/features_movies_test.csv'

features_df_train = None
features_df_test = None

if domain == 'apps':
    features_df_train = pd.read_csv(train_apps, index_col=0)
    features_df_test = pd.read_csv(test_apps, index_col=0)
elif domain == 'movies':
    features_df_train = pd.read_csv(train_filmes, index_col=0)
    features_df_test = pd.read_csv(test_filmes, index_col=0)
else:
    print("Verifique domínio, saindo")
    exit()

metodos = ['naive_bayes', 'svm', 'tree', 'nn']

if method not in metodos:
    print("Método inválido... Saindo")
    exit()

name_model = "%s_%s.model" % (domain, method)
name_result = "%s_%s.json" % (domain, method)

# exit()
# python script_classification.py domain method model/method+domain result/method+domain

# for file_path in dominios:
# features_df_train = pd.read_csv(train_apps, index_col=0)
# features_df_test = pd.read_csv(test_apps, index_col=0)
# features_df_train = pd.read_csv(train_filmes, index_col=0)
# features_df_test = pd.read_csv(test_filmes, index_col=0)

# print(features_df_train.columns)
# limit_test = 30


# CONTAR CLASSES
# Divide by class
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

train_normalized = Classification.normalize(df_train_under.iloc[:, 0:-1])

count_class_0, count_class_1 = features_df_test.helpfulness.value_counts()
df_class_0 = features_df_test[features_df_test['helpfulness'] == 0]
df_class_1 = features_df_test[features_df_test['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
##
# df_class_0_under = df_class_0.sample(limit_test)
# df_class_1_under = df_class_1.sample(limit_test)
# df_test_under = pd.concat([df_class_0_under, df_class_1_under], axis=0)
##
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

# df_test_under = pd.read_pickle('test_t.pkl')

# print(df_test_under.head())
test_normalized = Classification.normalize(df_test_under.iloc[:, 0:-1])
# print(test_normalized[:5])

classificador = Classification(
    method, train_normalized, df_train_under['helpfulness'], test_normalized, df_test_under['helpfulness'])

# pp = pprint.PrettyPrinter(indent=4)
# print('filmes')
resultado = classificador.classifier(model_folder+name_model)
with open(result_folder+name_result, 'w') as f:
    json.dump(resultado, f)

# pprint.pprint(resultado)
