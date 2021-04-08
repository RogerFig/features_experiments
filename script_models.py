# run tudo
# Pearson 0 - 1
# Spearman: -1 1
# Corr Pandas
from review_lenght_features import calc_features
from readability_features import calc_read_features
from spelling_features import calc_features_spell
from revrank_features import calc_revrank_feature
from aspects_features import calc_aspect_features
from subjectivity_features import calc_liwc, count_sentiment_divergence, calc_percent_subjectivity
from syntatic_features import calculate_percents
from sent_diver_features import calc_star_divergence
from tf_idf_features import get_tfidf

from read_corpus import load_corpus
import pandas as pd
from utils import Preprocessing
import numpy as np
import gzip

base = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/'
# Preprocessing
file_path_apps_dev = base+'dev_apps.pkl'
file_path_apps_train = base+'train_apps.pkl'
file_path_apps_test = base+'test_apps.pkl'

file_path_movies_dev = base+'dev_filmes.pkl'
file_path_movies_train = base+'train_filmes.pkl'
file_path_movies_test = base+'test_filmes.pkl'

dominios = {'apps_dev': file_path_apps_dev, 'movies_dev': file_path_movies_dev, 'apps_test': file_path_apps_test,
            'movies_test': file_path_movies_test, 'apps_train': file_path_apps_train, 'movies_train': file_path_movies_train}

for desc, file_path in dominios.items():
    # resultado_correlacoes = []  # feature, pearson, spearman
    df = load_corpus(file_path)
    df = df.astype({'stars': 'float64'})
    # df = df.head()

    domain = df['domain'][0]
    # print(domain)
    # print(df.columns)
    # print(df)
    p = Preprocessing()
    text_pre = df['text'].apply(p.preprocessing)
    text_pre = text_pre.to_frame(name='tokens')
    text_pre.to_pickle('preprocessed/%s_tokens.pkl.gzip' % desc)

    df_pre = pd.concat([df, text_pre], axis=1)
    df_pre = df_pre[df_pre.tokens.str.len() > 0]

    # df_features_final = pd.DataFrame
    # Lexical Features
    df_lex = calc_features(df_pre)
    # print(df_lex)

    # Readability
    df_readability = calc_read_features(df_pre)
    # print(df_readability)

    # Spelling errors
    df_spell = calc_features_spell(df_pre)
    # print(df_spell)

    # rev_rank
    df_rev_rank = calc_revrank_feature(df_pre)
    # print(df_rev_rank)

    # aspects
    df_aspects = calc_aspect_features(df_pre, domain)
    # print(df_aspects)

    # subjectivity

    df_liwc = calc_liwc(df_pre)
    # print(df_liwc)

    df_divergence = count_sentiment_divergence(df_pre)
    # print(df_divergence)

    df_subjectivity = calc_percent_subjectivity(df_pre)
    # print(df_subjectivity)

    df_syntatic = calculate_percents(df_pre)
    # print(df_syntatic)

    df_star_deviation = calc_star_divergence(df_pre)
    # print(df_star_deviation)

    df_star_helpfulness = df_pre[['stars', 'helpfulness']]
    df_all = pd.concat([df_lex, df_readability, df_spell, df_rev_rank, df_aspects,
                        df_liwc, df_divergence, df_subjectivity, df_syntatic, df_star_deviation, df_star_helpfulness], axis=1)

    df_all.to_csv('features/features_%s.csv' % desc)

    # tfidf_features = get_tfidf(df_pre)
    # f = gzip.GzipFile('features/tfidf_%s.npy' % desc, "w")
    # np.save(f, tfidf_features)
    # f.close()
