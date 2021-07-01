#########################################################
#########################################################
#####################    TWITTER    #####################
#########################################################
####  in case of twitter API interrupting, this code ####
# still works (there are few try-exception in the code) #
#########################################################

import tweepy as tw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import numpy as np
import psycopg2
from nltk.corpus import stopwords
import re
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from .. import UTILS_FOLDER_PATH

## PARAMS
TOPIC_MODELLING_CLASSFIER_PATH = UTILS_FOLDER_PATH+"topic_model.pickle"
TFIDF_PATH = UTILS_FOLDER_PATH+"tfidf.pickle"

consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"


# TWITTER authentication :
try:
    print('Connecting to TWITTER API ...')
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
except:
    print("Cannot connect to TWITTER API !")


# read the data (composed from our recipes steps and a foreign text)
def get_data():
    # load the dataset
    other_data = open('recipes/scripts/topic_modelling/corpus.txt').readlines()
    print(f"other data count: {len(other_data)}")

    conn = psycopg2.connect(
        host="157.230.24.228",
        database="cookix_db",
        user="cookix_user_db",
        password="f9d6UVP6gxEqueopMCiKdpjC0A5Pi5Ww",
    )
    cursor = conn.cursor()

    cursor.execute("SELECT steps FROM recipes_recipe;")
    recipes_steps = cursor.fetchall()
    recipes_steps = [steps[0] for steps in recipes_steps]
    recipes_steps = [steps for steps in recipes_steps if steps.strip() != ""]
    print(f"steps recipes count: {len(recipes_steps)}")

    cursor.execute("SELECT ingredients FROM recipes_recipe;")

    # get all recipes ingredients
    recipes_ingredients = cursor.fetchall()
    recipes_ingredients = [ings[0] for ings in recipes_ingredients]
    recipes_ingredients = [ings for ings in recipes_ingredients if ings.strip() != ""]
    print(f"ingredients recipes count: {len(recipes_ingredients)}")

    x_data = other_data + recipes_steps + recipes_ingredients
    y_data = list(np.zeros((len(other_data)), dtype="int")) + list(
        np.ones((len(recipes_steps) + len(recipes_ingredients)), dtype="int"))

    return x_data, y_data


def clean_text(sentences):

    processed_sen = []
    stop_words = set(stopwords.words('english'))

    for sen in sentences:
        # keep only text
        sentence = re.sub(r"[^a-z, ]+", ' ', sen.lower())
        sentence = re.sub(r'\b\w\b', ' ', sentence.strip())
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        # remove stopwords
        #sen_tokens = [w for w in sentence.split(" ")]
        #sentence = " ".join(sen_tokens)
        processed_sen.append(sentence)

    return processed_sen


# transform text sentences to TF IDF
def transform_to_tfidf(data):
    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    xdata_tfidf = tfidf.fit_transform(data)

    print("output shape (sentences_count, features_count): ", xdata_tfidf.shape)

    with open(TFIDF_PATH, 'wb') as file:
        pickle.dump(tfidf, file, protocol=pickle.HIGHEST_PROTOCOL)

    return xdata_tfidf


# train the topic
def train_topic_model_classifier(classifier_name=TOPIC_MODELLING_CLASSFIER_PATH):
    x_data, y_data = get_data()
    processed_data = clean_text(x_data)
    tfidf_data = transform_to_tfidf(processed_data)

    classifier = svm.SVC()
    classifier.fit(tfidf_data, y_data)
    with open(classifier_name, 'wb') as file:
        pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)

    return classifier


## TOPIC MODELLING CLASSIFIER
def filter_recipes_topic(sentences):

    tfidf_sentences = tfidf.transform(sentences).toarray()
    predictions = classifier.predict(tfidf_sentences)
    cooking_topic_sentences = [sen for sen, pred in zip(sentences, predictions) if pred == 1]

    return cooking_topic_sentences


# return positive tweets percentage and count
def get_users_feedbacks(keywords, num_items=100, sentiment_pct = 0.01):

    try:
        tweets = tw.Cursor(api.search,
                           q=keywords,
                           lang="en",
                           since='2020-11-01').items(num_items)

        all_tweets = [tweet.text for tweet in tweets]
        # remove duplicated tweets
        all_tweets = clean_text(all_tweets)
        all_tweets = list(set(all_tweets))
        cooking_sentences = filter_recipes_topic(all_tweets)

        sentiment_analyser = SentimentIntensityAnalyzer()
        pos_sen = 0
        neg_sen = 0
        for sen in cooking_sentences:
            result = sentiment_analyser.polarity_scores(sen) # returns ex: {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            if result['pos'] > sentiment_pct: pos_sen += 1
            elif result['neg'] > sentiment_pct: neg_sen += 1

        count = pos_sen + neg_sen
        print(count)
        if count > 0:
            return int((pos_sen / count) * 100), pos_sen
        else:
            return 0, 0

    except:
        return 0, 0


if not os.path.isfile(TOPIC_MODELLING_CLASSFIER_PATH):
    train_topic_model_classifier()
else:
    print("loading topic modeling classifier ...")
    with open(TFIDF_PATH, 'rb') as file:
        tfidf = pickle.load(file)

    with open(TOPIC_MODELLING_CLASSFIER_PATH, 'rb') as file:
        classifier = pickle.load(file)
