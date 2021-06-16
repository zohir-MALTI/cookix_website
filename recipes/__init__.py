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
from .scripts import script
import time

print(script.HELLO)

######################################################
##############    RECIPES GENERATION    ##############
######################################################

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

## params
INDEX_TO_WORD_FILE = "word_to_index.pickle"
STEPS_BEGINNING_FILE = "steps_beginning.pickle"
# MODEL_NAME = "gen_weights_202106121306_seq5_4647sen_771vocab_ep500.h5"
# MODEL_NAME = "gen_weights_with_202106132054_seq4_1359839sen_14902vocab_ep100.h5"
MODEL_NAME = "gen_weights_202106150755_seq4_1519332sen_11763vocab_ep100.h5"
INPUT_SEQ_LENGTH = 5

print('Loading spacy model ...')
nlp = spacy.load('en_core_web_lg')

def preprocess_text(sentences: list):
    processed_sen = []
    all_words = []

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for sen in sentences:

        if not ("Ingredients" in sen or "Save Recipe" in sen or "Print Recipe" in sen):
            # replace the pipe "|" that we used for steps separation, by space
            sentence = re.sub(r"\|", ' ', sen)
            # replace numbers by [number] (valid for 5, 5555, 55-5555 ....)
            sentence = re.sub(r"\s+[0-9]+(\-[0-9])*\s+", ' __NUMBER__ ', sentence.lower())
            # remove brackets
            sentence = re.sub(r'\(.+\)', ' ', sentence)
            # remove meaningless comments
            sentence = re.sub(r'Serves.*', ' ', sentence)
            # remove meaningless comments
            sentence = re.sub(r'Serves.*', ' ', sentence)

            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            sen_tokens = word_tokenize(sentence)
            sen_tokens = [w for w in sen_tokens if not w in stop_words]
            sentence = [lemmatizer.lemmatize(w) for w in sen_tokens]
            [all_words.append(w) for w in sentence]
            processed_sen.append(sentence)

    return processed_sen, all_words


def get_data():

    conn = psycopg2.connect(
        host="157.230.24.228",
        database="cookix_db",
        user="cookix_user_db",
        password="f9d6UVP6gxEqueopMCiKdpjC0A5Pi5Ww",
    )

    cursor = conn.cursor()
    cursor.execute("SELECT steps FROM recipes_recipe;")
    data = cursor.fetchall()
    data = [steps[0] for steps in data]

    clean_data = [steps for steps in data if steps.strip() != ""]
    clean_data = clean_data

    return clean_data


def get_encoded_sentences():

    data = get_data()
    processed_sen, all_words = preprocess_text(data)

    n_words = len(all_words)
    unique_words = list(set(all_words))
    n_unique_words = len(unique_words)

    word_to_index, index_to_word = {}, {}
    for i, w in enumerate(unique_words):
        index_to_word[i] = w
        word_to_index[w] = i

    input_sequence = []
    output_words = []
    steps_beginning_idx = []
    input_seq_length = 5

    for sen in processed_sen:
        for i in range(len(sen) - input_seq_length):
            in_seq = sen[i:i + input_seq_length]
            out_seq = sen[i + input_seq_length]
            in_seq_index = [word_to_index[word] for word in in_seq]
            input_sequence.append(in_seq_index)
            output_words.append(word_to_index[out_seq])
            if i == 0 and in_seq_index not in steps_beginning_idx:
                steps_beginning_idx.append(in_seq_index)

    X = np.reshape(input_sequence, (len(input_sequence), input_seq_length, 1))
    # X = X / float(vocab_size)
    # y = to_categorical(output_words)
    y = np.array(output_words.copy())
    y = np.reshape(y, (y.shape[0], 1))
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # save our variables

    with open(INDEX_TO_WORD_FILE, 'wb') as file:
        pickle.dump(index_to_word, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(STEPS_BEGINNING_FILE, 'wb') as file:
        pickle.dump(steps_beginning_idx, file, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y, n_unique_words


def train_model_generator():

    X, y, n_unique_words = get_encoded_sentences()

    EPOCHS = 50
    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = "adam"

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(32))
    # model.add(Dropout(0.1))
    # model.add(LSTM(32))
    model.add(Dense(n_unique_words, activation='softmax'))

    model.summary()

    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    current_time = datetime.today().strftime("%Y%m%d%H%M")
    TEXT_GEN_WEIGHTS_NAME = f"gen_weights_{current_time}_seq{INPUT_SEQ_LENGTH}_{X.shape[0]}sen_{n_unique_words}vocab_ep{EPOCHS}"

    print(TEXT_GEN_WEIGHTS_NAME)
    checkpoint = ModelCheckpoint(TEXT_GEN_WEIGHTS_NAME + ".hdf5", monitor='loss', verbose=1, save_best_only=True,
                                 mode='min')

    model.fit(X, y, batch_size=512, epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

    model.load_weights(TEXT_GEN_WEIGHTS_NAME + ".hdf5")
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    os.remove(TEXT_GEN_WEIGHTS_NAME + ".hdf5")
    model.save(TEXT_GEN_WEIGHTS_NAME + ".h5")

    return model


def get_sentences_beginning():

    with open(INDEX_TO_WORD_FILE, 'rb') as file:
        index_to_word = pickle.load(file)

    with open(STEPS_BEGINNING_FILE, 'rb') as file:
        steps_beginning_idx = pickle.load(file)

    steps_beginning_dict = {}
    for sen in steps_beginning_idx:
        sentence_text = [index_to_word[w] for w in sen]
        sentence_text = ' '.join(sentence_text)
        steps_beginning_dict[sentence_text] = sen

    steps_beginning_dict = OrderedDict(sorted(steps_beginning_dict.items(), key=lambda t: t[0]))

    # random_seq_index = np.random.randint(0, len(steps_beginning_idx) - 1)
    # random_seq = steps_beginning_idx[random_seq_index]
    #
    # word_sequence = [index_to_word[value] for value in random_seq]
    # print(f"Random sentence: '{random_sen}'")
    return steps_beginning_dict


def generate_sentences(model, input_sentence: list, steps_count: int=10):

    # model = load_model(MODEL_NAME + ".h5")

    with open(INDEX_TO_WORD_FILE, 'rb') as file:
        index_to_word = pickle.load(file)




    input_sentence_list = input_sentence.copy()
    word_to_index = {v: k for k, v in index_to_word.items()}

    stop_words = set(stopwords.words('english'))
    stop_words_idx = [word_to_index[w] for w in stop_words if w in word_to_index.keys()]

    sentences_delimiter = word_to_index["."]
    generated_sen_count = input_sentence_list.count(sentences_delimiter)
    input_seq_length = len(input_sentence_list) # model.input_shape[1]
    # print("AAAA: ", input_sentence_list)
    # print("AAAA: ", )

    words_count = 0
    # print("AAAA: ", input_sentence_list)
    # print("AAAA: ", type(input_sentence_list))
    # print("AAAA: ", input_seq_length)
    # print("AAAA: ", model.input_shape)
    while generated_sen_count <= steps_count:
        input_sentence_arr = np.reshape(input_sentence_list[-input_seq_length:], (1, input_seq_length, 1))
        # print("BBBBBB: ", input_sentence_arr)
        # print("BBBBBB: ", type(input_sentence_arr))

        predicted_word_index = model.predict(input_sentence_arr, verbose=0)
        # print(predicted_word_index)
        # print(predicted_word_index.shape)
        # predicted_word_id = np.argmax(predicted_word_index)
        predicted_idx_asc = np.argsort(predicted_word_index, axis=1, kind='quicksort')
        # predicted_word_id = np.argmax(predicted_word_index)
        # print("PPPPPPP: ", predicted_word_id)
        predicted_word_id = predicted_idx_asc[:,-1][0]
        # print("PPPPPPP: ", input_sentence_list)
        # print("LIST : ", np.reshape(input_sentence_arr, (input_sentence_arr.shape[1])))
        # print("LIST : ", input_sentence_list)

        # avoid words repetition
        # if the generated word is not a stopwords and appears among the N last words, do not add it
        # print("ZZZZZZZZZ: ", len(input_sentence_arr))
        print("MMMMMMIN : ", min(len(input_sentence_list), 15))
        keeped_length_words = min(len(input_sentence_list), 15)
        # print("NNNNNNN : ", keeped_length_words)
        # last_N_words = np.reshape(input_sentence_arr, (input_sentence_arr.shape[1]))[-keeped_length_words:]
        last_N_words = input_sentence_list[-keeped_length_words:]
        print("ZZZZZZZZZ: ", input_sentence_list)
        print("ZZZZZZZZZ: ", last_N_words)
        # print(input_sentence_list)
        # print(len(input_sentence_list))
        print("LAST WORDS : ", [index_to_word[w] for w in last_N_words] )
        print("PRED1 : ", index_to_word[predicted_word_id])
        # print(bb[-3:])
        # print(input_sentence_arr[,-4])
        # break
        i = 2
        while predicted_word_id not in stop_words_idx and predicted_word_id in last_N_words\
                and predicted_word_id not in [word_to_index["."], word_to_index[","]] :
            # print("TTTTTTTTTTTTTTTTT")
            predicted_word_id = predicted_idx_asc[:, -i][0]
            print("PRED2 : ", index_to_word[predicted_word_id])
            print("IIII: ", i)
            i += 1

        print("==================")
        input_sentence_list.append(predicted_word_id)
        if predicted_word_id == sentences_delimiter : generated_sen_count += 1
        words_count += 1
        if words_count >= 250: break

    words_seq = [index_to_word[index] for index in input_sentence_list]
    final_output = " ".join(words_seq)

    return final_output


def get_ing_keywords(output_name="ingredients_keywords.pickle"):

    if not os.path.isfile(output_name):
        print("getting ingredients keywords (this operation could take time)....")
        conn = psycopg2.connect(
            host="157.230.24.228",
            database="cookix_db",
            user="cookix_user_db",
            password="f9d6UVP6gxEqueopMCiKdpjC0A5Pi5Ww",
        )

        cursor = conn.cursor()
        cursor.execute("SELECT ingredients FROM recipes_recipe;")
        data = cursor.fetchall()
        data = [ing[0] for ing in data]
        clean_data = [ing for ing in data if ing.strip() != ""]
        clean_data = clean_data[:500]

        ingredients = []
        tt = time.time()
        for sen in clean_data:

            sentence = re.sub(r"[^a-z, ]+", ' ', sen.lower())
            sentence = re.sub(r'\b\w\b', ' ', sentence.strip())
            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            sentence_parts = [sen.strip() for sen in sentence.split(",")]
            for sen in sentence_parts:
                doc = nlp(sen)
                if len(doc) > 0:
                    if len(doc) > 1:
                        if doc[-1].pos_ == "NOUN" and doc[-2].pos_ == "NOUN":
                            ingredients.append(doc[-2:].text)
                    elif doc[-1].pos_ == "NOUN":
                        ingredients.append(doc[-1].text)

        print(len(ingredients))
        ingredients = list(set(ingredients))
        print(len(ingredients))
        print(f"TTTTIME: {time.time() - tt}")
        with open(output_name, 'wb') as file:
            pickle.dump(ingredients, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print("BBBBBBBBBBBBBBBBBBIIII")
        with open(output_name, 'rb') as file:
            ingredients = pickle.load(file)

    return ingredients

print("Loading generator model ...")
if MODEL_NAME is None:
    generator_model = train_model_generator()
else:
    generator_model = load_model(MODEL_NAME)
    INGREDIENTS_KEYWORDS = get_ing_keywords()
    # model = Sequential()
    # model.load_weights(MODEL_NAME)
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

print()

#####################################################
###################    TWITTER    ###################
#####################################################

print('Connecting to TWITTER API ...')
consumer_key    = "n1s4JvfETvz0hv8xsZxextI4K"
consumer_secret = "C1yHFjCW6ZIu3BjV9L5vj2huCEZW2jK14SQHkkxyXDx7RSmUf1"
access_key      = "1367830484173066243-iiTH7gTAP7xiRVIAkk8zObE0q0d3xu"
access_secret   = "b8MWjtlO52sEA5cgsoy4CfcS4nPKs5ar9x3yHDd1agBPE"

# Authentification :
# auth = tw.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)
# api = tw.API(auth, wait_on_rate_limit=True)

# print("Done !")

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
