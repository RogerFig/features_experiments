import numpy as np
from pprint import pprint
import sys
import pandas as pd
from classification import Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import joblib
import os
import json

domain = sys.argv[1]
model_folder = sys.argv[2]
# result_folder = sys.argv[4]
# sel_features = sys.argv[5]

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

train_normalized = Classification.normalize(df_train_under.iloc[:, 0:-1])

count_class_0, count_class_1 = features_df_test.helpfulness.value_counts()
df_class_0 = features_df_test[features_df_test['helpfulness'] == 0]
df_class_1 = features_df_test[features_df_test['helpfulness'] == 1]
df_class_0_under = df_class_0.sample(count_class_1)

df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)


test_normalized = Classification.normalize(df_test_under.iloc[:, 0:-1])


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
    clf, random_grid, scoring="f1", n_jobs=-1, min_resources="exhaust", factor=3)

halving_cv.fit(train_normalized, df_train_under['helpfulness'])

joblib.dump(halving_cv.best_estimator_, os.sep.join(
    [model_folder, 'rf_model_grid_%s.model' % (domain)]))

best = halving_cv.best_score_
best_params = halving_cv.best_params_

with open(os.sep.join(
        [model_folder, 'rf_model_grid_%s.params' % (domain)]), 'w') as f:
    f.write(str(best)+'\n')
    f.write(str(best_params)+'\n')
