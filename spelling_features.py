from read_corpus import load_corpus
from nltk.tokenize import word_tokenize


def get_vocab():
    vocab = []
    with open('vocab/vocab.txt') as f:
        for linha in f.readlines():
            vocab.append(linha.strip())
    return vocab


def get_spelling_errors(text, vocab):
    '''
        Total Spelling Errors
    '''
    total = 0
    words = word_tokenize(text)
    for word in words:
        if word.lower() not in vocab:
            print(word)
            total+=1
    return total

if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    vocab = get_vocab()
    # print(df.columns)
    for ind, row in df.iterrows():
        print("%d, Spelling: %d;" % (ind, get_spelling_errors(row['text'], vocab)))
        # print("Total Palavra: %d" % get_num_words(row[1]['text']))
        # print("Média: %d" % get_avg_sent_len(row[1]['text']))