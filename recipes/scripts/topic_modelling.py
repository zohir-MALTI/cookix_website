#################################################################
#################################################################
#####################    TOPIC MODELLING    #####################
#################################################################
#################################################################


import tweepy as tw
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import tensorflow
import psycopg2
import re
from datetime import datetime
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from collections import OrderedDict
import time
import itertools
# print(recipe_generator.HELLO)


print('Connecting to TWITTER API ...')
consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"

# Authentification :
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)

print("loading topic modeling classifier ...")
with open("", 'rb') as file:
    index_to_word = pickle.load(file)

# print("Done !")

