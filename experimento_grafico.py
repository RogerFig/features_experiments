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
import concurrent.futures
import sys
import json


def count_simi(parametros):
    '''
    i: index of document
    simi: similarity array of document i
    T: list of threshold similarity
    utilidade: array of document helpfulness
    '''
    i, simi, T, utilidade = parametros
    total_pares_simi = 0
    cont_iguais = 0
    candidates = np.where(simi >= T)[0]
    maiores = candidates[np.where(candidates > i)[0]]
    list_equals = []
    if len(maiores) > 0:
        slice_h = utilidade[maiores]
        total_pares_simi = len(slice_h)
        iguais = maiores[np.where(slice_h == utilidade[i])[0]]
        cont_iguais = len(iguais)
        list_equals = iguais.tolist()
        # print(cont_iguais)
    return (total_pares_simi, cont_iguais, list_equals, T)

# 83 sec - 10k
# Quantidade de pares em que houve similaridade acima de um threshold e tinham a mesma utilidade
# Quantos pares tinham a mesma utilidade considerando um determinado threshold de similaride?
# Cnp = n!\p!*(n-p)!
# [2134238050, 454297095, 82375945, 15192815, 2522693, 619359, 370352, 185585, 142727]

# TODO: 1. dividir em tamanhos de opiniões
#           2. listar similares para avaliação posterior


base = sys.argv[1]
# base = '/home/rogerio/workspace/corpus_splited/'
# base = '/home/rfsousa/workspace/corpus_splited/'
# base = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/'

# Pickles
file_path_apps_train = base+'dev_apps.pkl'
file_path_movies_train = base+'dev_filmes.pkl'

# Tokens
pre_apps_dev = 'preprocessed/apps_dev_tokens.pkl.gzip'
pre_filmes_dev = 'preprocessed/movies_dev_tokens.pkl.gzip'

dominios = {'dev_filmes': [
    file_path_movies_train, pre_filmes_dev], 'dev_apps': [file_path_apps_train, pre_apps_dev]}

for dominio, files in dominios.items():
    corpus = files[0]
    tokens = files[1]

    df = load_corpus(corpus)
    df_p = pd.read_pickle(tokens)
    df = df.astype({'stars': 'float64'})

    text_pre = df_p['tokens'].apply(lambda x: [a.lower() for a, _, _ in x])
    text_pre = text_pre.to_frame(name='tokens')

    df_pre = pd.concat([text_pre, df['helpfulness']], axis=1)
    df_pre = df_pre[df_pre.tokens.str.len() > 0]
    # print(df_pre[df_pre.tokens.str.len() > 30]['helpfulness'].value_counts())
    # df_pre = pd.read_pickle('de_test.pkl')

    # df_pre = df_pre.sample(frac=0.01)
    # print(df_pre.pivot_table(index=['helpfulness'], aggfunc='size'))

    # QUEBRAR EM DUAS PARTES TAMANHO 30
    df_pre_short = df_pre[df_pre.tokens.str.len() <= 30]
    df_pre_long = df_pre[df_pre.tokens.str.len() > 30]

    subsets = {'short': df_pre_short, 'long': df_pre_long}

    for title_set, subset in subsets.items():
        dct = Dictionary(subset['tokens'])
        corpus = [dct.doc2bow(line)
                  for ind, line in subset['tokens'].iteritems()]
        index_temp = get_tmpfile("index")
        index = Similarity(index_temp, corpus, num_features=len(dct))

        len_docs = len(corpus)
        totais = []
        utilidade = subset['helpfulness'].values

        cont_iguais = [0 for i in range(0, 9)]
        total_pares_simi = [0 for i in range(0, 9)]
        T = [k/10 for k in range(1, 10)]

        dict_equals = {k/10: {} for k in range(1, 10)}

        futures = []
        for i, simi in enumerate(index):
            print('%s/%s' % (i, len_docs), end='\r')
            param_list = [(i, simi, T[l], utilidade) for l in range(9)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(count_simi, params)
                           for params in param_list]
            # for k in range(9):
            #     candidates = np.where(simi >= T[k])[0]
            #     maiores = np.where(candidates > i)[0]
            #     if len(maiores) > 0:
            #         slice_h = utilidade[maiores]
            #         total_pares_simi[k] += len(slice_h)
            #         iguais = np.where(slice_h == utilidade[i])
            #         cont_iguais[k] += len(iguais[0])
            for ind, f in enumerate(futures):
                resultado = f.result()
                total_pares_simi[ind] += resultado[0]
                cont_iguais[ind] += resultado[1]
                threshold = resultado[3]
                dict_equals[threshold][i] = resultado[2]

        for i in range(9):
            if total_pares_simi[i] != 0:
                percentual = cont_iguais[i]/total_pares_simi[i]
                totais.append(percentual)
            else:
                totais.append(0)
        result = pd.DataFrame(
            [totais], columns=[10, 20, 30, 40, 50, 60, 70, 80, 90])
        result.to_csv('exp_similaridades/%s_%s_ttgrafico.csv' %
                      (dominio, title_set))
        json.dump(dict_equals, open('exp_similaridades/%s_%s_ids_reviews_simi.json' %
                                    (dominio, title_set), 'w'))

# fim = time.time()
# print("Tempo: %s" % (fim-inicio))
# 0,44771,8352,1916,546,127,62,45,11,10 -> 1000
# 0,364,74,17,3,1,0,0,0,0 -> 100
