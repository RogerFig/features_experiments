from read_corpus import load_corpus
# from nltk.tokenize import word_tokenize
from review_lenght_features import calc_features
from textstat import textstat
import pandas as pd


def get_ari(text):
    '''
    Automated Readability Index
    '''

    textstat.set_lang('pt_BR')
    return textstat.automated_readability_index(text)


def get_gunning_fog(text):
    '''
    Gunning Fog Index
    '''

    textstat.set_lang('pt_BR')
    return textstat.gunning_fog(text)


def get_flesch_index(text):
    '''
    Flesch Reading Ease Index
    '''

    textstat.set_lang('pt_BR')
    return textstat.flesch_reading_ease(text)


def get_coleman_index(text):
    '''
    Coleman-Liau Index
    '''

    textstat.set_lang('pt_BR')
    return textstat.coleman_liau_index(text)


def get_flesch_kincaid_index(text):
    '''
    Flesch-Kincaid Grade Level
    '''

    textstat.set_lang('pt_BR')
    return textstat.flesch_kincaid_grade(text)


def get_SMOG(text):
    '''
    Simple Measure of Gobbledygook Score
    '''

    textstat.set_lang('pt_BR')
    return textstat.smog_index(text)


def calc_read_features(documents):
    features_table = []
    for ind, row in documents.iterrows():
        ARI = get_ari(row['text'])
        GF = get_gunning_fog(row['text'])
        FI = get_flesch_index(row['text'])
        CI = get_coleman_index(row['text'])
        FKI = get_flesch_kincaid_index(row['text'])
        SMOG = get_SMOG(row['text'])
        features_table.append([ind, ARI, GF, FI, CI, FKI, SMOG])
    df = pd.DataFrame(features_table, columns=[
        'ind', 'ARI', 'GF', 'FI', 'CI', 'FKI', 'SMOG'])
    idx = pd.Index(df['ind'], name='')
    df.index = idx
    df.drop(['ind'], axis=1, inplace=True)
    return df


if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    df = df.head()
    print(calc_read_features(df))
