# -*- coding: utf-8 -*-

from os import stat
import numpy as np
from nltk.tokenize import word_tokenize
from read_corpus import load_corpus
import getpass
import re

import nlpnet
from nltk import RSLPStemmer
from nltk.corpus import stopwords

# from enelvo import normaliser

"""
Preprocess and creates a graph structure
"""

nlpnet.set_data_dir('/home/'+getpass.getuser()+'/pos-pt/')


class Preprocessing:

    def __init__(self):
        """
        Constructor
        """
        # self.norm = normaliser.Normaliser()
        self.pos_tagger = nlpnet.POSTagger()
        self.stemmer = RSLPStemmer()

    @staticmethod
    def remover_caracteres_especiais(palavra):
        """
        Remover caracteres especiais
        :param palavra
        :return: palavra apenas com números, letras e espaco
        Unicode normalize transforma um caracter em seu equivalente em latin.
        """
        #nfkd = unicodedata.normalize('NFKD', palavra)
        #palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

        # Usa expressão regular para retornar a palavra apenas com números, letras e espaço
        # return re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)
        return re.sub('[^a-zA-Z0-9 \\\]', '', palavra)

    def stem_word(self, word):
        return self.stemmer.stem(word)

    # If necessary
    def remove_emoji(self, review):
        """
        Remove emojis from reviews
        :param review:
        :return: reviews whithout emojis
        """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        n_review = emoji_pattern.sub(r'', review)
        emoji_pattern = re.compile(
            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
            u"(\ud83c[\udde0-\uddff])|"
            u"([\u2600-\u27bf])"  # flags (iOS)
            "+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', n_review)

    def tagger(self, review):
        """
        Runs nlpnet tagger and keeps only relevant words
        :param review:
        :return: relevant words
        """
        openclass_tags = ['V', 'N', 'ADJ', 'ADV', 'PROSUB',
                          'NPROP', 'PROADJ', 'PROPESS', 'PCP', 'IN']
        tagged_words = []

        for elements in self.pos_tagger.tag(review):
            for word, tag in elements:
                if tag in openclass_tags:
                    tagged_words.append((word, tag, self.stem_word(word)))
        return tagged_words

    def normalize_review(self, review):
        """
        Normalizes repeated characters: kkkkkkk -> k, muuuuiiiitttooo -> muito
        :param review: a review
        :return: a normalized review
        """
        return self.norm.normalise(review)

    def preprocessing(self, review):
        """
        Preprocess the reviews
        :return: a list of words
        """

        # NOVOS REVIEWS
        texto = review
        texto = self.remove_emoji(texto)
        #normalized_sentence = self.normalize_review(texto)

        words = self.tagger(texto)

        words = [
            (w, t, s) for w, t, s in words if w not in stopwords.words(u'portuguese')]
        return words


if __name__ == '__main__':

    #reivews_path = 'reviews/'
    #files = os.listdir(reivews_path)
    rev = {'texto': 'A casa é amarela e azul com bolinhas vermelhas. Muito obrigado!'}
    p = Preprocessing()
    print(p.preprocessing(rev))
    # for file in files:
    #    print(file)
    #    p = Preprocessing(reivews_path+file)
    #    print(p.tagger('A casa é amarela.'))
    #    exit()
