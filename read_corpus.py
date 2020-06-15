import pandas as pd
import os
import ast

def load_corpus(path_corpus):
    if path_corpus.endswith('.pkl'):
        df = pd.read_pickle(path_corpus)
        return df
    elif path_corpus.endswith('.csv'):
        df = pd.read_csv(path_corpus)
        return df
    else:
        return load_corpus_folders(path_corpus)



def load_corpus_folders(path_corpus="corpus"):
    #root = 'reviews/corpus'

    print("Loading...")
    revs = [os.path.join(path, nome) for path, _, nomes in os.walk(path_corpus) for nome in nomes if nome.endswith('.json')]

    lista_out = []
    for p in revs:
        with open(p) as rev:
            lista_out.append([p.split('/')[1],ast.literal_eval(rev.read())])

    df_out = pd.DataFrame(lista_out,columns=['domain','review'])
    return df_out