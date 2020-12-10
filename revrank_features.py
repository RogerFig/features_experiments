from numpy import core
from utils import Preprocessing
from numpy.core.fromnumeric import argsort
from read_corpus import load_corpus
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import numpy as np
from nltk import FreqDist
import nltk

# Lembrete: O vetor é criado para cada produto.
# A lista de frequencia externa não posssui acentuação, verificar isso.


def get_freqs_nilc():
    map_out = {}
    with open('resources/formas.saocarlos.txt', errors='ignore') as f:
        freq_list = f.readlines()

    for linha in freq_list:
        linha_sp = linha.split()
        try:
            map_out[linha_sp[1]] = int(linha_sp[0])
        except IndexError:
            pass
    return map_out


def get_dominant_words(freqs, freq_external, limit):
    words = []
    vet = []
    for k in freqs.keys():
        if k in freq_external and math.log(freq_external[k], 2) > 0:
            vet.append(freqs[k] * 3 * (1/math.log(freq_external[k])))
            words.append(k)
    vet = np.array(vet)
    args_ord = np.argsort(vet)
    args_ord = args_ord[::-1]
    args_ord = args_ord[:limit]

    words = [words[i] for i in args_ord]

    return (words, args_ord)


def map_reviews(words, core_vector):
    rev_vectors = []
    for texto in words:
        rev_vectors.append(
            [1 if word in texto else 0 for word in core_vector])
    return rev_vectors


def calc_S(words, features, core_vector, c=20):
    vec_core = [1 for i in range(len(core_vector))]
    vr = np.dot(features, vec_core)
    tam = []
    for doc in words:
        tam.append(len(doc))
    media = sum(tam)/len(tam)
    punitive = [1/c if a > media else 1 for a in tam]
    S = punitive * (vr/tam)
    return S


def calc_revrank_feature(documents, limit=200):
    fd = FreqDist()
    words = []
    for _, row in documents.iterrows():
        t = [w for w, _, _ in row['tokens']]
        words.append(t)
        fd.update(t)

    freq_external = get_freqs_nilc()
    dominant = get_dominant_words(fd, freq_external, limit)
    features = map_reviews(words, dominant[0])
    S = calc_S(words, features, dominant[0])
    df = pd.DataFrame(S, index=documents.index, columns=['S'])
    return df


# if __name__ == "__main__":
#     p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'

#     # stopwords = nltk.corpus.stopwords.words('portuguese')

#     df = load_corpus(p)
#     df = df.head()

#     p = Preprocessing()
#     tokenized = df['text'].apply(p.preprocessing)

#     words = tokenized.apply(
#         lambda x: [a.lower() for a, _, _ in x])

#     fd = FreqDist()

#     for _, rev in words.iteritems():
#         fd.update(rev)

#     freq_external = get_freqs_nilc()

#     resultado = get_dominant_words(fd, freq_external, 10)

#     features = map_reviews(words, resultado[0])

#     vcfv = np.array([1 for i in range(len(resultado[0]))])

#     calc_S(words, features, resultado[0])
