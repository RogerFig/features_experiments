from utils import Preprocessing
from read_corpus import load_corpus
import pandas as pd
import liwc
from collections import Counter


def parse_token(token, parse, lexicon):
    bruto = lexicon.get(token, [])
    bib = [cat for cat in parse(token)]
    if len(bruto) > len(bib):
        return bruto
    else:
        return bib


def calc_liwc(documents):
    lexicon, _ = liwc.read_dic("resources/LIWC2007_Portugues_win_UTF8.dic")
    parse, category_names = liwc.load_token_parser(
        "resources/LIWC2007_Portugues_win_UTF8.dic")
    categories = ['negate', 'swear', 'affect',
                  'posemo', 'negemo', 'anx', 'anger', 'sad']
    counted = []
    for id, doc in documents.iterrows():
        counted.append(dict(
            Counter([category for token, _, _ in doc['tokens'] for category in parse_token(token, parse, lexicon)])))
    table = []
    for dic in counted:
        aux = [dic.get(cat, 0) for cat in categories]
        table.append(aux)
    df = pd.DataFrame(table, columns=categories, index=documents.index)
    return df


def load_sentilex():
    sentilex = {}
    with open('resources/sentilex-reduzido.txt') as f:
        for line in f:
            sp = line.strip().split(',')
            sentilex[sp[0].strip()] = int(sp[1])

    return sentilex

# Calcular 2 valores:
# - Polarity Force
# - Polaridade de cada Review


def count_sentiment_divergence(documents):
    sentilex = load_sentilex()
    sentiment_force = []
    count_pos = count_neg = 0
    feature_divergence = []
    for id, row in documents.iterrows():
        doc = row['tokens']
        pol = dict(Counter([sentilex.get(s, 0) for s, _, _ in doc]))
        pos = (pol.get(1, 0) - pol.get(-1, 0))/len(doc)
        neg = (pol.get(-1, 0) - pol.get(1, 0))/len(doc)
        force = abs(pol.get(1, 0) - pol.get(-1, 0))/len(doc)
        if pos > 0.02:
            sentiment_force.append([id, 'pos', force])
            count_pos += 1
        elif neg > 0.015:
            sentiment_force.append([id, 'neg', force])
            count_neg += 1
        else:
            sentiment_force.append([id, 'neu', force])
    if count_pos > count_neg:
        f_revs = [f for _, p, f in sentiment_force if p == 'pos']
        media = sum(f_revs)/len(f_revs)
        feature_divergence = [f-media for _, _, f in sentiment_force]
    elif count_pos < count_neg:
        f_revs = [f for _, p, f in sentiment_force if p == 'neg']
        media = sum(f_revs)/len(f_revs)
        feature_divergence = [f-media for _, _, f in sentiment_force]
    else:
        print('O improvável aconteceu')

    df = pd.DataFrame(feature_divergence, columns=[
                      'divergence'], index=documents.index)
    return df


def calc_percent_subjectivity(documents):
    sentilex = load_sentilex()
    feature_subjectivity = []
    for id, doc in documents.iterrows():
        pol = dict(Counter([sentilex.get(s, 0) for s, _, _ in doc['tokens']]))
        sub = pol.get(1, 0) + pol.get(-1, 0)
        obj = pol.get(0, 0)
        feature_subjectivity.append([sub/(sub+obj)])
    df = pd.DataFrame(feature_subjectivity, columns=[
                      'subjectivity'], index=documents.index)
    return df


# p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
# df = load_corpus(p)
# df = df.head()
# p = Preprocessing()
# preprocessed_p1 = df['text'].apply(p.preprocessing)
# preprocessed_p1 = preprocessed_p1.apply(
#     lambda x: [a.lower() for a, _, _ in x])
# # print(calc_liwc(preprocessed_p1))
# # print(count_sentiment_words(preprocessed_p1))
# print(calc_percent_subjectivity(preprocessed_p1))
