from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from read_corpus import load_corpus
import pandas as pd
from utils import Preprocessing
import numpy as np

base = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/'
file_path_apps_dev = base+'dev_apps.pkl'
pre_apps_dev = 'preprocessed/apps_dev_tokens.pkl.gzip'

df = load_corpus(file_path_apps_dev)
df_p = pd.read_pickle(pre_apps_dev)
df = df.astype({'stars': 'float64'})


# p = Preprocessing()
# text_pre = df['text'].apply(p.preprocessing)
text_pre = df_p['tokens'].apply(lambda x: ' '.join([a for a, _, _ in x]))
text_pre = text_pre.to_frame(name='tokens')

df_pre = pd.concat([text_pre, df['helpfulness']], axis=1)
df_pre = df_pre[df_pre.tokens.str.len() > 0]
df_pre = df_pre.iloc[:100]
# print(df_pre)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_pre['tokens'])
# X = X / np.sum(X, axis=1)
# print(X)
cos = cosine_similarity(X, X)

# print(df_pre.iloc[0, 1])

# print(cos)

totais = []
for i in range(1, 10):
    T = i/10
    cont = 0
    for i in range(len(df_pre)):
        for j in range(i, len(df_pre)):
            if i != j:
                if cos[i, j] > T:
                    if df_pre.iloc[i, 1] == df_pre.iloc[j, 1]:
                        # print("1: ", df_pre.iloc[i, 0])
                        # print("2: ", df_pre.iloc[j, 0])
                        cont += 1
    totais.append(cont)
result = pd.DataFrame([totais], columns=[10, 20, 30, 40, 50, 60, 70, 80, 90])
result.to_csv('dev_apps_grafico.csv')
