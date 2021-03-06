####################################################################
#####################    RECIPES GENERATION    #####################
####################################################################
####################################################################
########  this script allows us train the generator model   ########
####################################################################

import spacy
import numpy as np
import psycopg2
import re
from datetime import datetime
import os
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from collections import OrderedDict
import itertools
#import gpt_2_simple as gpt2

# local imports
from .. import UTILS_FOLDER_PATH


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

## params
INDEX_TO_WORD_FILE = UTILS_FOLDER_PATH+"word_to_index.pickle"
STEPS_BEGINNING_FILE = UTILS_FOLDER_PATH+"steps_beginning.pickle"
MODEL_NAME = UTILS_FOLDER_PATH+"gen_weights_202106151706_seq10_1457145sen_11759vocab_ep100.h5"
INPUT_SEQ_LENGTH = 5

print('Loading spacy model ...')
nlp = spacy.load('en_core_web_lg')


# to clean sentences
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


# encode text sentences to numbers
# ex: mix tomatoes , onion and garlic in a bowl
# with a sentence length of 4, we will have:
#        mix tomatoes , onion => and (target)
#        tomatoes , onion and => garlic (target)
#        , onion and garlic => in (target)
#        ......
def get_encoded_sentences():

    data = get_data()
    processed_sen, all_words = preprocess_text(data)

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

    # get the xtrain and ytrain
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


# generator trainign (LSTM)
def train_model_generator():

    X, y, n_unique_words = get_encoded_sentences()

    EPOCHS = 50
    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = "adam"

    model = Sequential()
    model.add(Embedding(n_unique_words, 200))
    model.add(LSTM(200))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(n_unique_words, activation='softmax'))

    model.summary()

    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    current_time = datetime.today().strftime("%Y%m%d%H%M")
    TEXT_GEN_WEIGHTS_NAME = f"gen_weights_{current_time}_seq{INPUT_SEQ_LENGTH}_{X.shape[0]}sen_{n_unique_words}vocab_ep{EPOCHS}"

    checkpoint = ModelCheckpoint(TEXT_GEN_WEIGHTS_NAME + ".hdf5", monitor='loss', verbose=1, save_best_only=True,
                                 mode='min')

    model.fit(X, y, batch_size=512, epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

    model.load_weights(TEXT_GEN_WEIGHTS_NAME + ".hdf5")
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    os.remove(TEXT_GEN_WEIGHTS_NAME + ".hdf5")
    model.save(TEXT_GEN_WEIGHTS_NAME + ".h5")

    return model


# recipes beginning extracted from our recipes
def get_sentences_beginning():

    with open(INDEX_TO_WORD_FILE, 'rb') as file:
        index_to_word = pickle.load(file)

    with open(STEPS_BEGINNING_FILE, 'rb') as file:
        steps_beginning_idx = pickle.load(file)

    steps_beginning_dict = {}
    for sen in steps_beginning_idx:
        sentence_text = [index_to_word[w] for w in sen]
        if sentence_text[0].strip() in ["adjust", "arrange", "blend", "boil", "bring", "coarsely",
                                        "coat", "cook", "cut", "fill", "first", "grind",
                                        "halve", "heat", "light", "make", "marinate", "melt", "mix", "peel",
                                        "place", "preheat", "put", "slice", "soak", "stir"]:

            sentence_text = ' '.join(sentence_text)
            steps_beginning_dict[sentence_text] = sen

    steps_beginning_dict = OrderedDict(sorted(steps_beginning_dict.items(), key=lambda t: t[0]))

    return steps_beginning_dict


# function that generates a recipe
def generate_sentences(model, input_sentence: list, steps_count: int=10):

    # model = load_model(MODEL_NAME + ".h5")

    with open(INDEX_TO_WORD_FILE, 'rb') as file:
        index_to_word = pickle.load(file)

    # encode the recipe beginning
    input_sentence_list = input_sentence.copy()
    word_to_index = {v: k for k, v in index_to_word.items()}

    stop_words = set(stopwords.words('english'))
    stop_words_idx = [word_to_index[w] for w in stop_words if w in word_to_index.keys()]

    sentences_delimiter = word_to_index["."]
    generated_sen_count = input_sentence_list.count(sentences_delimiter)
    input_seq_length = len(input_sentence_list) # model.input_shape[1]

    # generate word by word without repetition
    words_count = 0
    while generated_sen_count <= steps_count:
        input_sentence_arr = np.reshape(input_sentence_list[-input_seq_length:], (1, input_seq_length, 1))
        predicted_word_index = model.predict(input_sentence_arr, verbose=0)
        predicted_idx_asc = np.argsort(predicted_word_index, axis=1, kind='quicksort')
        predicted_word_id = predicted_idx_asc[:, -1][0]


        # avoid words repetition
        # if the generated word is not a stopwords and appears among the N last words, do not add it
        keeped_length_words = min(len(input_sentence_list), 20)
        last_N_words = input_sentence_list[-keeped_length_words:]
        i = 2
        while predicted_word_id not in stop_words_idx and predicted_word_id in last_N_words \
                and predicted_word_id not in [word_to_index["."], word_to_index[","]] :
            predicted_word_id = predicted_idx_asc[:, -i][0]
            i += 1

        input_sentence_list.append(predicted_word_id)
        if predicted_word_id == sentences_delimiter : generated_sen_count += 1
        words_count += 1
        if words_count >= 250: break

    words_seq = [index_to_word[index] for index in input_sentence_list]
    final_output = " ".join(words_seq)

    return final_output


# get all available keywords for recipes without words duplicate
def get_ing_keywords(output_name=UTILS_FOLDER_PATH+"ingredients_keywords.pickle"):

    if not os.path.isfile(output_name):
        print("getting ingredients keywords (this operation could take around 7 minutes)....")
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
        clean_data = clean_data

        ingredients = []
        for sen in clean_data:

            sentence = re.sub(r"[^a-z, ]+", ' ', sen.lower())
            sentence = re.sub(r'\b\w\b', ' ', sentence.strip())
            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            ingredients.append(sentence)

        ingredients = [sen for sen in ingredients if sen.strip()!=""]
        ings_parts = []
        for sentence in ingredients:
            ings_parts.append([sen.strip() for sen in sentence.split(",") if sen.strip()!=""])

        ings_parts = list(itertools.chain(*ings_parts)) ## flat the list
        ings_parts = list(set(ings_parts))

        ingredients = []
        for sen in ings_parts:
            doc = nlp(sen)
            if len(doc) > 1:
                if doc[-1].pos_ == "NOUN" and doc[-2].pos_ == "NOUN":
                    ingredients.append(doc[-2:].text)
            elif doc[-1].pos_ == "NOUN":
                ingredients.append(doc[-1].text)

        ingredients = list(set(ingredients))

        # remove duplicates ["oil", "oi", "olive oil", "olive"] => ["olive oil"]
        ings_lengths_dict = {}
        for w in ingredients:
            ings_lengths_dict[w] = len(w)

        ings_lengths_dict = OrderedDict(sorted(ings_lengths_dict.items(), key=lambda t: t[1]))
        ings_lengths_dict = list(ings_lengths_dict.keys())
        ings_without_duplicates = ings_lengths_dict.copy()
        for w in ings_lengths_dict:
            temp_list = ings_lengths_dict.copy()
            temp_list.remove(w)
            for w_temp in temp_list:
                if w in w_temp:
                    ings_without_duplicates.remove(w)
                    break

        # save the object, to load it after
        with open(output_name, 'wb') as file:
            pickle.dump(ings_without_duplicates, file, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(output_name, 'rb') as file:
            ings_without_duplicates = pickle.load(file)

    return ings_without_duplicates

"""
def load_GPT2():
    # gpt2.download_gpt2(model_name="124M")
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=UTILS_FOLDER_PATH+'run1')

    gpt2.generate(sess,
                  length=250,
                  temperature=0.7,
                  prefix="mix in a bowl",
                  nsamples=5,
                  batch_size=5
                  )

"""

print("Loading generator model ...")
if MODEL_NAME is None:
    generator_model = train_model_generator()
else:
    generator_model = load_model(MODEL_NAME)

INGREDIENTS_KEYWORDS = get_ing_keywords()
