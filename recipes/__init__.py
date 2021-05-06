import tweepy as tw
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print('Connecting to TWITTER API ...')
consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"
# Authentification :
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)

print('Loading spacy model ...')
nlp = spacy.load('en_core_web_lg')

print("Done !")

class TwitterStats:

    def __clean_text(self, text):
        # remove accents and set everything lowercase
        preprocessor = CountVectorizer(strip_accents="ascii", token_pattern=r"\w+", lowercase=True).build_preprocessor()

        if pd.isnull(text) or type(text) != str:
            return None
        else:
            return preprocessor(text)


    def __lemmatize_text(self, text, drop_stop=True, drop_punctuation=True, list_tag=None):
        # Lemmatize, remove stopwords and puntuation and keep some tag
        doc = nlp(text)
        lemma = []
        for token in doc:
            if list_tag is None:
                if (token.is_stop != drop_stop) and (token.is_punct != drop_punctuation) and (token.text not in string.punctuation):
                    lemma.append(token.lemma_)
            else:
                if (token.is_stop != drop_stop) and (token.is_punct != drop_punctuation) and (token.tag_ in list_tag) and (token.text not in string.punctuation):
                    lemma.append(token.lemma_)
        if len(lemma) > 0:
            return lemma
        else:
            return None


    def __nlp_pipeline(self, text: str):
        text = text.replace('\n', ' ').replace('\r', '')
        text = ' '.join(text.split())
        text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
        text = re.sub(r"(\s\-\s|-$)", "", text)
        text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
        text = re.sub(r"\&\S*\s", "", text)
        text = re.sub(r"(\&|\+|\#|\$|\£|\%|\:|\@|\-)", "", text)

        return text


    def get_tweets(self, keywords, num_items = 1000):

        tweets = tw.Cursor(api.search,
                           q=keywords,
                           lang="en",
                           since='2021-04-01').items(num_items)
        all_tweets = [self.__nlp_pipeline(tweet.text) for tweet in tweets]
        # all_tweets = [tweet.text for tweet in tweets]

        return all_tweets


    def analyze(self, keywords, pct = 0.4):
        sentences = self.get_tweets(keywords, num_items=100)
        sentiment_analyser = SentimentIntensityAnalyzer()
        pos_sen = []
        neg_sen = []
        for sentence in sentences:
            result = sentiment_analyser.polarity_scores(sentence)
            if result['pos'] > pct: pos_sen.append(sentence)
            if result['neg'] > pct: neg_sen.append(sentence)
            # print("ss: ", ss)
            # for k in sorted(ss):
            #     print('{0}: {1}, '.format(k, ss[k]), end='')
            # print()

        print("POS:")
        print(pos_sen)
        print("NEG:")
        print(neg_sen)
        print("CCCCCCCCCCCCCC")
        count = len(pos_sen) + len(neg_sen)
        print("CCCCCCCCCCCCCC")
        if count > 0:
            return round(len(pos_sen) / count, 2), round(len(neg_sen) / count, 2)
        else:
            return 0, 0
