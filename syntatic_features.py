from utils import Preprocessing
from read_corpus import load_corpus
import pandas as pd


def count_tags(review):
    tags = ['N', 'V', 'PCP', 'ADV', 'ADJ']
    count_tags = {'N': 0, 'V': 0, 'PCP': 0, 'ADV': 0, 'ADJ': 0, 'total': 0}
    for _, tag, _ in review:
        count_tags['total'] += 1
        if tag in tags:
            count_tags[tag] += 1
    return count_tags


def calculate_percents(documents):
    all_counts = []

    for ind, review in documents.iterrows():
        all_counts.append(count_tags(review['tokens']))

    counts = []

    for cc in all_counts:
        aux = []
        aux.append(cc['N'])
        aux.append(cc['V'] + cc['PCP'])
        aux.append(cc['ADV'])
        aux.append(cc['ADJ'])
        total = sum(aux)

        if cc['total'] == 0:
            aux.append(0)
            aux = [0 for x in aux]
            counts.append(aux)
            continue

        aux = [x/cc['total'] for x in aux]

        aux.append(total/cc['total'])

        counts.append(aux)

    df = pd.DataFrame(
        counts, columns=['nouns', 'verbs', 'adv', 'adj', 'open'], index=documents.index)
    return df


# p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
# df = load_corpus(p)
# r = calculate_percents(df['text'])
# print(r)
