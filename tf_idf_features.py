from read_corpus import load_corpus
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import Preprocessing
from gensim import corpora
from gensim import models


def get_tfidf_old(documents):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(documents)
    features_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=features_names)
    return df


def get_tfidf(documents):
    dictionary = corpora.Dictionary(documents)
    bow_corpus = [dictionary.doc2bow(text)
                  for _, text in documents.iteritems()]
    tfidf = models.TfidfModel(bow_corpus)

    return(tfidf[bow_corpus])
# print(tst2)


if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    # vocab = get_vocab()
    # p1 = df[(df.object == 'com.google.android.apps.books')
    # & (df.helpfulness == 1.0)]
    p = Preprocessing()

    part1 = df[df.helpfulness == 0.0]
    # part1 = part1.head()
    preprocessed_p1 = part1['text'].apply(p.preprocessing)

    preprocessed_p1 = preprocessed_p1.apply(
        lambda x: [a.lower() for a, _, _ in x])

    # corpus_tfidf = get_tfidf(preprocessed_p1)
    # tams = [len(rev) for rev in corpus_tfidf]
    # tams.sort()
    # print(tams)
    pu = preprocessed_p1.apply(lambda x: ' '.join([a for a in x]))

    tfidf_p1 = get_tfidf_old(pu)

    ord = tfidf_p1.iloc[0].sort_values(ascending=False)
    print(ord.iloc[:13])
    # print(p1.iloc[0]['text'])
