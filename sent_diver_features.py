from read_corpus import load_corpus
import pandas as pd
import numpy as np


def calc_star_divergence(documents):
    documents = documents.astype({'stars': 'float64'})
    media = documents.groupby('object')['stars'].mean()
    dev = np.array([])
    for obj, mean in media.iteritems():
        result = documents[documents.object == obj]['stars'] - mean
        dev = np.append(dev, result.values)

    df = pd.DataFrame(dev, columns=['stars_deviation'], index=documents.index)
    return df


# p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
# df = load_corpus(p)
# print(len(calc_sentiment_divergence(df)))
