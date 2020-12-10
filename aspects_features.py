from utils import Preprocessing
from read_corpus import load_corpus
import pandas as pd


def count_aspects(documents, aspects_list, domain):
    '''
        @param documents: lista de comentários lematizados ou stemizados
        @param aspects_list: lista de aspectos lematizados ou stemizados
    '''
    contagem = documents.apply(lambda x: len(
        [a for _, _, a in x if a in aspects_list[domain]]))

    return contagem


def load_aspect_list():
    p = Preprocessing()
    aspect_list = {'apps': [], 'movies': []}
    apps = []
    movies = []
    with open('resources/aspects_apps.txt') as f:
        for linha in f:
            apps.append(p.preprocessing(linha.strip())[0])

    with open('resources/aspects_movies.txt') as f:
        for linha in f:
            movies.append(p.preprocessing(linha.strip())[0])
    aspect_list['apps'] = [s for _, _, s in apps]
    aspect_list['movies'] = [s for _, _, s in movies]
    return aspect_list


def calc_aspect_features(documents, domain):
    aspect_list = load_aspect_list()
    contagem = documents['tokens'].apply(lambda x: len(
        [a for _, _, a in x if a in aspect_list[domain]]))
    contagem = contagem.to_frame(name='aspects')
    return contagem


# p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
# df = load_corpus(p)
# p = Preprocessing()
# preprocessed_p1 = df['text'].apply(p.preprocessing)

# preprocessed_p1 = preprocessed_p1.apply(
#     lambda x: [a.lower() for _, _, a in x])

# aspect_list = []
# with open('resources/aspects_apps.txt') as f:
#     for linha in f:
#         aspect_list.append(p.preprocessing(linha.strip())[0])

# aspect_list = [s for _, _, s in aspect_list]

# print(count_aspects(preprocessed_p1, aspect_list))
