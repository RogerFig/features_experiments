from read_corpus import load_corpus
from nltk.tokenize import word_tokenize


def get_avg_sent_len(text):
    '''
    Tamanho médio de sentenças
    '''
    # sents = text.split('.')
    return get_num_words(text) / get_num_sents(text)
    

def get_num_sents(text):
    '''
    Quantidade de Sentenças
    '''
    sents = text.split('.')
    return len(sents)


def get_num_words(text):
    '''
    Quantidade de palavras
    '''
    words = word_tokenize(text)
    return len(words)


if __name__ == "__main__":
    p = 'corpus'
    df = load_corpus(p)
    for ind, row in df.iterrows():
        print("%d, Total Sentenças: %d; Total Palavra: %d; Média: %d " % (ind, get_num_sents(row[1]['text']), get_num_words(row[1]['text']), get_avg_sent_len(row[1]['text'])))
        # print("Total Palavra: %d" % get_num_words(row[1]['text']))
        # print("Média: %d" % get_avg_sent_len(row[1]['text']))