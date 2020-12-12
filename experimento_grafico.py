from liwc import dic
from pandas.core.arrays.sparse import dtype
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from read_corpus import load_corpus
import pandas as pd
# from utils import Preprocessing
import numpy as np
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities
from gensim.test.utils import datapath, get_tmpfile
from gensim.similarities import Similarity

# 83 sec - 10k
# Quantidade de pares em que houve similaridade acima de um threshold e tinham a mesma utilidade
# Quantos pares tinham a mesma utilidade considerando um determinado threshold de similaride?
# Cnp = n!\p!*(n-p)!
# [2134238050, 454297095, 82375945, 15192815, 2522693, 619359, 370352, 185585, 142727]

base = '/home/rogerio/workspace/corpus_splited/'
# base = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/'

# Pickles
file_path_apps_train = base+'train_apps.pkl'
file_path_movies_train = base+'train_filmes.pkl'

# Tokens
pre_apps_dev = 'preprocessed/apps_train_tokens.pkl.gzip'
pre_filmes_dev = 'preprocessed/movies_train_tokens.pkl.gzip'

dominios = {'train_apps': [file_path_apps_train, pre_apps_dev], 'train_filmes': [
    file_path_movies_train, pre_filmes_dev]}

for dominio, files in dominios.items():
    corpus = files[0]
    tokens = files[1]

    df = load_corpus(corpus)
    df_p = pd.read_pickle(tokens)
    df = df.astype({'stars': 'float64'})

    # p = Preprocessing()
    # text_pre = df['text'].apply(p.preprocessing)
    # text_pre = df_p['tokens'].apply(lambda x: ' '.join([a for a, _, _ in x]))
    text_pre = df_p['tokens'].apply(lambda x: [a.lower() for a, _, _ in x])
    text_pre = text_pre.to_frame(name='tokens')

    df_pre = pd.concat([text_pre, df['helpfulness']], axis=1)
    df_pre = df_pre[df_pre.tokens.str.len() > 0]

    # df_pre = df_pre.sample(frac=0.01)
    # print(df_pre.pivot_table(index=['helpfulness'], aggfunc='size'))
    dct = Dictionary(df_pre['tokens'])
    corpus = [dct.doc2bow(line) for ind, line in df_pre['tokens'].iteritems()]
    index_temp = get_tmpfile("index")
    index = Similarity(index_temp, corpus, num_features=len(dct))

    # for sims in index:
    #     print(sims)
    #     break

    # query = next(iter(corpus))
    # model = TfidfModel(corpus, smartirs='txx')
    # corpus_tfidf = model[corpus]
    # print(dir(corpus_tfidf))
    # print(dir(model))
    # index = similarities.MatrixSimilarity(corpus_tfidf)
    # index.save('/tmp/deerwester.index')
    # index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    # sims = index[corpus_tfidf[1]]
    # print(sims.nonzero())
    # print(dct)
    # print(model)

    # vectorizer = CountVectorizer(dtype=np.int64)
    # X = vectorizer.fit_transform(df_pre['tokens'])

    # cos = cosine_similarity(X, X, dense_output=False)

    # print(cos.toarray())
    # a, b = cos.nonzero()
    # print(a.dtype)

    len_docs = len(corpus)
    totais = []
    utilidade = df_pre['helpfulness'].values
    for k in range(1, 10):
        T = k/10
        cont_iguais = 0
        # print(k)
        total_pares_simi = 0
        for i, simi in enumerate(index):
            print('%s/%s' % (i, len_docs), end='\r')
            candidates = np.where(simi >= T)[0]
            maiores = np.where(candidates > i)[0]
            if len(maiores) > 0:
                slice_h = utilidade[maiores]
                total_pares_simi += len(slice_h)
                iguais = np.where(slice_h == utilidade[i])
                cont_iguais += len(iguais[0])
            # indices = simi.nonzero()[0]
            # for j in candidates:
            #     if i < j:
            #         if utilidade[i] == utilidade[j]:
            #             cont += 1
        if total_pares_simi != 0:
            percentual = cont_iguais/total_pares_simi
            totais.append(percentual)
        else:
            totais.append(0)
    print(totais)
    result = pd.DataFrame(
        [totais], columns=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    result.to_csv('exp_similaridades/%s_grafico.csv' % dominio)

    # 0,44771,8352,1916,546,127,62,45,11,10 -> 1000
    # 0,364,74,17,3,1,0,0,0,0 -> 100
