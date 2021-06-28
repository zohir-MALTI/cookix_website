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


TOPIC_MODELLING_CLASSFIER_PATH = UTILS_FOLDER_PATH+"topic_model.pickle"
TFIDF_PATH = UTILS_FOLDER_PATH+"tfidf.pickle"

print('Connecting to TWITTER API ...')
consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"

# Authentification :
try:
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
except:
    print("Cannot connect to TWITTER API !")


# def __clean_text(text):
#     # remove accents and set everything lowercase
#     preprocessor = CountVectorizer(strip_accents="ascii", token_pattern=r"\w+", lowercase=True).build_preprocessor()
#
#     if pd.isnull(text) or type(text) != str:
#         return None
#     else:
#         return preprocessor(text)


# def __lemmatize_text(text, drop_stop=True, drop_punctuation=True, list_tag=None):
#     # Lemmatize, remove stopwords and puntuation and keep some tag
#     print('Loading spacy model ...')
#     nlp = spacy.load('en_core_web_lg')
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




# def __nlp_pipeline(text: str):
#     text = text.replace('\n', ' ').replace('\r', '')
#     text = ' '.join(text.split())
#     text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
#     text = re.sub(r"(\s\-\s|-$)", "", text)
#     text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
#     text = re.sub(r"\&\S*\s", "", text)
#     text = re.sub(r"(\&|\+|\#|\$|\£|\%|\:|\@|\-)", "", text)
#
#     return text



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
        sen_tokens = [w for w in sentence.split(" ") if w not in stop_words]
        sentence = " ".join(sen_tokens)
        processed_sen.append(sentence)

    return processed_sen


def transform_to_tfidf(data):
    tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    xdata_tfidf = tfidf.fit_transform(data)

    print("output shape (sentences_count, features_count): ", xdata_tfidf.shape)

    with open(TFIDF_PATH, 'wb') as file:
        pickle.dump(tfidf, file, protocol=pickle.HIGHEST_PROTOCOL)

    return xdata_tfidf


def train_topic_model_classifier(classifier_name = TOPIC_MODELLING_CLASSFIER_PATH):
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
    # print(predictions)
    cooking_topic_sentences = [sen for sen, pred in zip(sentences, predictions) if pred == 1]
    # print(cooking_topic_sentences)

    return cooking_topic_sentences


# return positive tweets percentage and count
def get_users_feedbacks(keywords, num_items=100, sentiment_pct = 0.5):

# try:
    print("AAAAAAAAAAAa")
    tweets = tw.Cursor(api.search,
                       q=keywords,
                       lang="en",
                       since='2021-04-01').items(num_items)
    # print(len(tweets))

    # urls = [tweet.id for tweet in tweets]
    # print(urls)
    all_tweets = [tweet.text for tweet in tweets]
    # print(dir([tweet for tweet in tweets][0]))
    # print([tweet.id for tweet in tweets])
    # print([tweet.id_str for tweet in tweets])
    # print("TWWWWWWWWWWWWWWW: ", all_tweets)
    all_tweets = clean_text(all_tweets)
    # remove duplicated tweets
    all_tweets = list(set(all_tweets))
    # print("TWWWWWWWWWWWWWWW: ", all_tweets)
    cooking_sentences = filter_recipes_topic(all_tweets)
    # all_tweets = [tweet.text for tweet in tweets]

    # sentences = self.get_tweets(keywords, num_items=100)
    sentiment_analyser = SentimentIntensityAnalyzer()
    pos_sen = 0
    neg_sen = 0
    for sen in cooking_sentences:
        result = sentiment_analyser.polarity_scores(sen) # returns ex: {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        # print(f"{result}   ==================> {sen}")
        if result['pos'] > sentiment_pct: pos_sen += 1
        elif result['neg'] > sentiment_pct: neg_sen += 1
        # print("ss: ", ss)
        # for k in sorted(ss):
        #     print('{0}: {1}, '.format(k, ss[k]), end='')
        # print()

    count = pos_sen + neg_sen
    if count > 0:
        return int((pos_sen / count) * 100), pos_sen
    else:
        return 0, 0

    # except:
    #     return 0, 0


if not os.path.isfile(TOPIC_MODELLING_CLASSFIER_PATH):
    train_topic_model_classifier()
else:
    print("loading topic modeling classifier ...")
    with open(TFIDF_PATH, 'rb') as file:
        tfidf = pickle.load(file)

    with open(TOPIC_MODELLING_CLASSFIER_PATH, 'rb') as file:
        classifier = pickle.load(file)
