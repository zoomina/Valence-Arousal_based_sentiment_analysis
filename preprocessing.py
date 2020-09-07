import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

class PreProcessing:
    def __init__(self, col):
        self.lang = 'ENG'
        self.col = col
        if self.lang == 'ENG':
            self.df = self.get_eng_data()
        # raise error
        assert type(self.col) is list, "col is not 'list' but {}".format(type(self.col))

    def get_lexicon(self):
        # Lexicon includes a list of more than 20,000 English words and their VAD scores.
        # For a given word and a demension(VAD), the scores range from 0 to 1.
        df_lexicon = pd.read_csv('data/ko-NRC-VAD-Lexicon.txt', sep='\t', encoding='utf-8', dtype={'Valence':np.float64, 'Arousal':np.float64, 'Dominance':np.float64})
        df_lexicon.columns = ['ENG', 'KOR', 'V', 'A', 'D']

        # data scaling : 0 to 1 -> -1 to 1
        df_lexicon[self.col] = (df_lexicon[self.col] - 0.5) * 2.

        return {key: df_lexicon[self.col].iloc[i].tolist() for i, key in enumerate(df_lexicon[self.lang])}

    def get_eng_data(self):
        # The range of VAD score is from 1 to 5 same to IEMOCAP dataset with the same scoring method, i.e. using 5-scales Self-Assessment Manikin(SAM)
        df_data = pd.read_csv('data/emobank.csv', encoding='utf-8', dtype={'V':np.float64, 'A':np.float64, 'D':np.float64})
        df_data.columns = ['id','split','V','A','D','ENG']

        # data scaling : 1 to 5 -> -1 to 1
        df_data[self.col] = (df_data[self.col] - 3.) / 2.
        print(df_data[self.col].max(), df_data[self.col].min())

        df_data[self.lang] = df_data.apply(lambda row: row[self.lang].lower(), axis=1)
        print('-- cleaning --\n', df_data.head())

        return df_data[['ENG']+self.col]

    def data_tokenizing(self):
        # tokenizing use nltk
        self.df[self.lang] = self.df.apply(lambda row: word_tokenize(row[self.lang]), axis=1)
        print('-- tokenize --\n', self.df.head())

    def remove_punctuation(self):
        # remove_punctuations
        def _remove_punc(wlist):
            for i, word in enumerate(wlist):
                if word == "n't":
                    continue
                wlist[i] = ''.join(c for c in word if c not in punctuation)
            wlist = [w for w in wlist if w != '']
            if not wlist:
                return np.nan
            return wlist

        self.df[self.lang] = self.df.apply(lambda row: _remove_punc(row[self.lang]), axis=1)
        self.df = self.df.dropna()
        print('-- remove punctuations --\n', self.df.head())

    def pos_tagger(self):
        self.df[self.lang] = self.df.apply(lambda row: pos_tag(row[self.lang]), axis=1)
        print('-- pos tagged --\n', self.df.head())

    def label_lexicon_score_to_data(self):
        lexicon = self.get_lexicon()
        def _label_score(sentence, lexicon):
            scored = {}
            if type(sentence[0]) == tuple:
                for word, tag in sentence:
                    if word not in lexicon.keys():
                        scored[word] = None
                        continue
                    scored[(word, tag)] = lexicon[word]
                return scored
            else:
                for word in sentence:
                    if word not in lexicon.keys():
                        scored[word] = None
                        continue
                    scored[word] = lexicon[word]
                return scored
        print(self.df[self.lang].head())
        self.df[self.lang] = self.df.apply(lambda row: _label_score(row[self.lang], lexicon), axis=1)

        print('\n==== labeled_score ====\n')
        print(self.df.head())

    def remove_stopwords_eng(self):
        def _get_removed(tokenized):
            stop_words = set(stopwords.words('english'))
            if type(list(tokenized.keys())[0]) == tuple:
                filtered_sentence = {word: val for word, val in tokenized.items() if(not word[0] in stop_words) or (val)}
            else:
                filtered_sentence = {word:val for word, val in tokenized.items() if (not word in stop_words) or (val)}

            return filtered_sentence

        self.df['ENG'] = self.df.apply(lambda row: _get_removed(row['ENG']), axis=1)
        print('-- remove stopwords --\n', self.df.head())

    def get_data(self, lexicon=True):
        self.data_tokenizing()
        self.remove_punctuation()
        self.pos_tagger()

        if lexicon:
            self.label_lexicon_score_to_data()
            self.remove_stopwords_eng()

        return self.df

def main():
    col = ['V','A']

    df_lex_data = PreProcessing(col).get_data(lexicon=True)
    df_lex_data.to_csv("data/preprocessed_lex.csv")

    df_none_data = PreProcessing(col).get_data(lexicon=False)
    df_none_data.to_csv("data/preprocessed_none.csv")

if __name__ == '__main__':
    main()