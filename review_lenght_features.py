from read_corpus import load_corpus
import pandas as pd
from utils import Preprocessing


def get_num_sents(text):
    '''
    Quantidade de Sentenças
    '''
    sents = text.split('.')
    return len(sents)


# def get_num_words(text):
#     '''
#     Quantidade de palavras
#     '''
#     return len(words)


def calc_features(documents):
    features_table = []
    for ind, row in documents.iterrows():
        num_words = len(row['tokens'])
        num_sents = get_num_sents(row['text'])
        avg_sent_len = num_words/num_sents
        features_table.append([ind, num_words, num_sents, avg_sent_len])
    df = pd.DataFrame(features_table, columns=[
                      'ind', 'num_words', 'num_sents', 'avg_sent_len'])
    idx = pd.Index(df['ind'], name='')
    df.index = idx
    df.drop(['ind'], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    df = df.head()
    # print(df)
    p = Preprocessing()
    pre = df['text'].apply(p.preprocessing)
    pre = pre.to_frame(name='tokens')
    c = pd.concat([df, pre], axis=1)
    # print(c)
    c = c[c.tokens.str.len() > 0]
    d = calc_features(c)
    print(d)
    # print(pre[pre.text == []])
    # print(pd.concat(df['text'], pre))
    # print(pre.columns)
