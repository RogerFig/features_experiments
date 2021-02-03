import pandas as pd
from classification import Classification
import pprint

dominios = ['features/features_apps_dev.csv', 'features/features_movies_dev']
train_apps = 'features/features_apps_train.csv'
test_apps = 'features/features_apps_test.csv'

train_filmes = 'features/features_movies_train.csv'
test_filmes = 'features/features_movies_test.csv'
# for file_path in dominios:
# features_df_train = pd.read_csv(train_apps, index_col=0)
# features_df_test = pd.read_csv(test_apps, index_col=0)
features_df_train = pd.read_csv(train_filmes, index_col=0)
features_df_test = pd.read_csv(test_filmes, index_col=0)

print(features_df_train.columns)

metodos = ['naive_bayes', 'svm', 'tree', 'nn']

# CONTAR CLASSES
# Divide by class
count_class_0, count_class_1 = features_df_train.helpfulness.value_counts()
df_class_0 = features_df_train[features_df_train['helpfulness'] == 0]
df_class_1 = features_df_train[features_df_train['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

train_normalized = Classification.normalize(df_train_under.iloc[:, 0:-1])

count_class_0, count_class_1 = features_df_test.helpfulness.value_counts()
df_class_0 = features_df_test[features_df_test['helpfulness'] == 0]
df_class_1 = features_df_test[features_df_test['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

test_normalized = Classification.normalize(df_test_under.iloc[:, 0:-1])

classificador = Classification(
    metodos[1], train_normalized, df_train_under['helpfulness'], test_normalized, df_test_under['helpfulness'])

# pp = pprint.PrettyPrinter(indent=4)
print('filmes')
pprint.pprint(classificador.classifier())
