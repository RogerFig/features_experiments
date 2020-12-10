from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from read_corpus import load_corpus
import pandas as pd
from utils import Preprocessing
import numpy as np

# base = '/home/rogerio/workspace/corpus_splited/'
base = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/'
file_path_apps_dev = base+'dev_filmes.pkl'
pre_apps_dev = 'preprocessed/movies_dev_tokens.pkl.gzip'

df = load_corpus(file_path_apps_dev)
df_p = pd.read_pickle(pre_apps_dev)
df = df.astype({'stars': 'float64'})


# p = Preprocessing()
# text_pre = df['text'].apply(p.preprocessing)
text_pre = df_p['tokens'].apply(lambda x: ' '.join([a for a, _, _ in x]))
text_pre = text_pre.to_frame(name='tokens')

df_pre = pd.concat([text_pre, df['helpfulness']], axis=1)
df_pre = df_pre[df_pre.tokens.str.len() > 0]
# df_pre = df_pre.iloc[:10]
# print(df_pre)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_pre['tokens'])
# X = X / np.sum(X, axis=1)
# print(X)
cos = cosine_similarity(X, X, dense_output=False)

# print(cos.toarray())
# a, b = cos.nonzero()
# print(a)
# print(b)
# print('zip: ', next(zip(a, b)))
# print(cos[a[1], b[1]])


totais = []
for i in range(1, 10):
    T = i/10
    cont = 0
    for i, j in zip(cos.nonzero()[0], cos.nonzero()[1]):
        if i > j:
            if cos[i, j] > T:
                if df_pre.iloc[i, 1] == df_pre.iloc[j, 1]:
                    # print("1: ", df_pre.iloc[i, 0])
                    # print("2: ", df_pre.iloc[j, 0])
                    cont += 1
    totais.append(cont)
result = pd.DataFrame([totais], columns=[10, 20, 30, 40, 50, 60, 70, 80, 90])
result.to_csv('dev_movies_grafico.csv')

# 0,44771,8352,1916,546,127,62,45,11,10 -> 1000
# 0,364,74,17,3,1,0,0,0,0 -> 100
