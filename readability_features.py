from read_corpus import load_corpus
from nltk.tokenize import word_tokenize
from textstat import textstat


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

if __name__ == "__main__":
    p = '/home/rogerio/workspace/Corpus Gigante/corpus_csvs_pickles/corpus_splited/dev_apps.pkl'
    df = load_corpus(p)
    print(df.columns)
    for ind, row in df.iterrows():
        print("%d, ARI: %d; G_FOG: %d; Flesch Ease: %d;" % (ind, get_ari(row['text']), get_gunning_fog(row['text']), get_flesch_index(row['text'])))
        # print("Total Palavra: %d" % get_num_words(row[1]['text']))
        # print("Média: %d" % get_avg_sent_len(row[1]['text']))