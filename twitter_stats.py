# https://stackoverflow.com/questions/13177428/instagram-api-to-fetch-pictures-with-specific-hashtags/16934513?fbclid=IwAR3E7llce13JDmQ5E0sePsLLpOPUzKvaBUkivrPjC0IP8YrEjAVAZo_vNy0#16934513
def clean_text(text):
    print(text)

# import numpy as np
# import os
# import tweepy as tw
# import pandas as pd
# import string
# from sklearn.feature_extraction.text import CountVectorizer
# import spacy
# import re
#
# consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
# consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
# access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
# access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"
#
# # Authentification :
# auth = tw.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)
# api = tw.API(auth, wait_on_rate_limit=True)
#
# requete = "couscous OR salmon OR Shrimp"
# tweets = tw.Cursor(api.search,
#                    q = requete,
#                    lang = "en",
#                    since='2021-04-01').items(100)
# all_tweets = [tweet.text for tweet in tweets]
#
#
# def nlp_pipeline(text):
#     text = text.replace('\n', ' ').replace('\r', '')
#     text = ' '.join(text.split())
#     text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
#     text = re.sub(r"(\s\-\s|-$)", "", text)
#     text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
#     text = re.sub(r"\&\S*\s", "", text)
#     text = re.sub(r"\&", "", text)
#     text = re.sub(r"\+", "", text)
#     text = re.sub(r"\#", "", text)
#     text = re.sub(r"\$", "", text)
#     text = re.sub(r"\£", "", text)
#     text = re.sub(r"\%", "", text)
#     text = re.sub(r"\:", "", text)
#     text = re.sub(r"\@", "", text)
#     text = re.sub(r"\-", "", text)
#
#     return text
#
#
# #remove accents and set everything lowercase
# preprocessor = CountVectorizer(strip_accents="ascii", token_pattern=r"\w+", lowercase=True).build_preprocessor()
# nlp = spacy.load('en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0')
#
# def clean_text2(text):
#     print(text)
#
# def clean_text(text):
#     if pd.isnull(text) or type(text) != str:
#         return None
#     else:
#         return preprocessor(text)
#
#
# def lemmatize_text(text, drop_stop=True, drop_punctuation=True, list_tag=None):
#     # Lemmatise, remove stopwords and puntuation and keep some tag
#     doc = nlp(text)
#     lemma = []
#     for token in doc:
#         if list_tag is None:
#             if (token.is_stop != drop_stop) and (token.is_punct != drop_punctuation) and (token.text not in string.punctuation):
#                 lemma.append(token.lemma_)
#         else:
#             if (token.is_stop != drop_stop) and (token.is_punct != drop_punctuation) and (token.tag_ in list_tag) and (token.text not in string.punctuation):
#                 lemma.append(token.lemma_)
#     if len(lemma) > 0:
#         return lemma
#     else:
#         return None
#
#
#
