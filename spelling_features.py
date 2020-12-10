from read_corpus import load_corpus
# from nltk.tokenize import word_tokenize
import pandas as pd


def get_vocab():
    vocab = []
    with open('vocab/vocab.txt') as f:
        for linha in f:
            vocab.append(linha.strip())
    return set(vocab)


def get_spelling_errors(text, vocab):
    '''
        Total Spelling Errors
    '''
    total = 0
    # words = word_tokenize(text)
    for word, _, _ in text:
        if word not in vocab:
            # print(word)
            total += 1
    return total


def calc_features_spell(documents):
    print('load vocab...')
    vocab = get_vocab()
    print('vocab loaded')
    features_table = []
    total = len(documents)
    for ind, row in documents.iterrows():
        print('%d/%d' % (ind, total), end='\r')
        spell_errors = get_spelling_errors(row['tokens'], vocab)
        features_table.append([ind, spell_errors])
    df = pd.DataFrame(features_table, columns=[
        'ind', 'spell'])
    idx = pd.Index(df['ind'], name='')
    df.index = idx
    df.drop(['ind'], axis=1, inplace=True)
    return df


# if __name__ == "__main__":
#     p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
#     df = load_corpus(p)
#     df = df.head()

    # vocab = get_vocab()
    # print(df.columns)
