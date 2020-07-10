from read_corpus import load_corpus
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    features_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=features_names)
    return df


if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    # vocab = get_vocab()
    # p1 = df[(df.object == 'com.google.android.apps.books')
    # & (df.helpfulness == 1.0)]
    p1 = df[df.helpfulness == 0.0]
    # print(p1['text'])
    tfidf_p1 = get_tfidf(p1['text'])
    ord = tfidf_p1.iloc[0].sort_values(ascending=False)
    print(ord.iloc[:13])
    print(p1.iloc[0]['text'])
