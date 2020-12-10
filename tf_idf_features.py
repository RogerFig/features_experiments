from read_corpus import load_corpus
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import Preprocessing
from gensim import corpora
from gensim import models
from collections import defaultdict
import numpy as np
import gzip


def get_tfidf_old(documents):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(documents)
    features_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=features_names)
    return df


def get_tfidf(documents):
    words = documents['tokens'].apply(lambda x: [w for w, _, _ in x])
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in words:
        for token in text:
            frequency[token] += 1

    words = [
        [token for token in text if frequency[token] > 1] for text in words
    ]
    dictionary = corpora.Dictionary(words)
    bow_corpus = [dictionary.doc2bow(text)
                  for text in words]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    features = np.zeros([len(bow_corpus), len(dictionary)])

    for i, doc in enumerate(corpus_tfidf):
        for pos, value in doc:
            features[i, pos] = value
    return features
# print(tst2)


if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    df = df[df.helpfulness == 1.0]
    p = Preprocessing()
    text_pre = df['text'].apply(p.preprocessing)
    text_pre = text_pre.to_frame(name='tokens')
    df_pre = pd.concat([df, text_pre], axis=1)
    df_pre = df_pre[df_pre.tokens.str.len() > 0]

    # pu = preprocessed_p1.apply(lambda x: ' '.join([a for a in x]))

    features = get_tfidf(df_pre)
    f = gzip.GzipFile('features/tfidf_%s.npy' % 'testando', "w")
    np.save(f, features)
    f.close()

    # To load
    # f = gzip.GzipFile('file.npy.gz', "r")
    # np.load(f)
    # print(df_pre.shape)
    # print(features.shape)
