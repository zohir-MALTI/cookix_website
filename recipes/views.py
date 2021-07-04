import random
import time
import numpy as np
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.postgres.search import *
from django.db.models import Q
import spacy
import re
from collections import Counter
import itertools
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
# local imports
from . import UTILS_FOLDER_PATH
from . import generate_sentences, generator_model, get_sentences_beginning, INGREDIENTS_KEYWORDS
from . import get_users_feedbacks
from .models import *

# MODEL NAMES & PARAMS
CLASSIFIER_NAME = UTILS_FOLDER_PATH+"recommendation_classifier.pickle"
RECIPES_INGREDIENTS_DF = UTILS_FOLDER_PATH+"recipes_by_ing_df.csv"
TOKEN_IN_TITLE_FACTOR = 10


def home(request):
    random_recipes = Recipe.objects.all()[4000:4009]
    # whether to display cluster
    user_id = request.user.id
    if user_id is not None and len(Likes.objects.all().filter(user_id=user_id)) > 0:
        clusters = custom_recipes_clusters(user_id)
    else:
        clusters = None
    return render(request, 'recipes/home.html', {'recipes': random_recipes,
                                                 'recipes_count': Recipe.objects.count(),
                                                 'todays_special': random_recipes[0],
                                                 'clusters': clusters})


def about_us(request):
    return render(request, 'recipes/about_us.html')


def custom_recipes_clusters(user_id, n_clusters=6, num_results=30):

    if not os.path.isfile(CLASSIFIER_NAME):
        print('Creating CLASSIFIER!')
        train_model()

    # load the classifier
    recipes_ing_df = pd.read_csv(RECIPES_INGREDIENTS_DF, sep='|', index_col=0)
    file = open(CLASSIFIER_NAME, 'rb')
    nearest_neighbors_algo = pickle.load(file)
    file.close()

    # get all liked recipe so that we can get their similar recipes
    pks = Likes.objects.all().filter(user_id=user_id)
    liked_recipes_idx = list([recipe.recipe_id.id for recipe in pks])

    similar_recipes_idx = []
    for recipe_id in liked_recipes_idx:
        recipe_array = recipes_ing_df.loc[int(recipe_id), :].to_numpy().reshape(1, -1)
        distances, nearest_idx = nearest_neighbors_algo.kneighbors(recipe_array, n_neighbors=num_results)
        real_idx_recipes = [recipes_ing_df.index[idx] for idx in nearest_idx[0]]
        similar_recipes_idx.append(real_idx_recipes)

    # remove duplicated recipes ids
    similar_recipes_idx = list(itertools.chain(*similar_recipes_idx))
    similar_recipes_idx = list(set(similar_recipes_idx))

    # get for each similar recipe, its array
    similar_recipes = []
    for recipe_id in similar_recipes_idx:
        recipe = list(recipes_ing_df.loc[int(recipe_id), :])
        recipe = [1 if x == TOKEN_IN_TITLE_FACTOR else x for x in recipe]
        similar_recipes.append(recipe)

    similar_recipes_arr = np.array(similar_recipes)

    # train a kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(similar_recipes_arr)
    labels = kmeans.labels_
    recipes_per_cluster = {}
    for cluster in range(n_clusters):
        cluster_idx = [idx for i, idx in enumerate(similar_recipes_idx) if labels[i] == cluster]
        # rename the cluster
        recipes_cluster_keywords = [recipe.title.strip().split(" ") for recipe in Recipe.objects.filter(pk__in=cluster_idx)]
        recipes_cluster_keywords = list(itertools.chain(*recipes_cluster_keywords))
        cluster_name = Counter(recipes_cluster_keywords).most_common(1)[0][0]
        recipes_per_cluster[cluster_name] = Recipe.objects.filter(pk__in=cluster_idx)

    return recipes_per_cluster

# get all the recipes for each diet
def diet_types(request, diet_type):
    if request.method == 'GET':
        page_number = request.GET.get("page")
        diet_type = diet_type.replace("-", " ")
        recipes = Recipe.objects.all().defer("ingredients", "summary")
        # switch case in python
        if diet_type == "vegetarian":
            recipes = recipes.filter(vegetarian="True")
        elif diet_type == "vegan":
            recipes = recipes.filter(vegan="True")
        elif diet_type == "vegan":
            recipes = recipes.filter(vegan="True")
        elif diet_type == "vegan":
            recipes = recipes.filter(vegan="True")
        elif diet_type == "vegan":
            recipes = recipes.filter(vegan="True")

        paginator = Paginator(recipes, 24)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': diet_type})


# get all the recipes for each dish type
def dish_types(request, dish_type):
    if request.method == 'GET':
        page_number = request.GET.get("page")
        dish_type = dish_type.replace("-", " ")
        recipes = Recipe.objects.all().defer("ingredients", "summary").filter(dish_types__icontains=dish_type)
        paginator = Paginator(recipes, 24)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': dish_type})


# get all the recipes for each cuisine type
def cuisine_types(request, cuisine_type):
    if request.method == 'GET':
        page_number = request.GET.get("page")
        cuisine_type = cuisine_type.replace("-", " ")
        recipes = Recipe.objects.all().defer("ingredients", "summary").filter(cuisines__icontains=cuisine_type)
        paginator = Paginator(recipes, 24)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': cuisine_type})


@login_required(login_url="/accounts/login")
def favorites(request):
    pks = Likes.objects.all().filter(user_id=request.user.id)
    pks = [recipe.recipe_id for recipe in pks]
    liked_recipes = Recipe.objects.filter(title__in=list(pks)).all()

    return render(request, 'recipes/favorites.html', {'recipes': liked_recipes})


# generic function to add like if exists, remove it otherwise
def add_action(request, recipe_likes_dislikes, recipe_id):
    if recipe_likes_dislikes.filter(user_id=request.user.id, recipe_id=recipe_id).exists():
        recipe_likes_dislikes.filter(user_id=request.user, recipe_id=get_object_or_404(Recipe, pk=recipe_id)).delete()
    else:
        recipe_likes_dislikes.create(user_id=request.user, recipe_id=get_object_or_404(Recipe, pk=recipe_id))


@login_required(login_url="/accounts/login")
def add_like(request, recipe_id):
    if request.method == 'POST':
        recipe_likes = Likes.objects
        add_action(request, recipe_likes, recipe_id)
        return redirect('/' + str(recipe_id))


@login_required(login_url="/accounts/login")
def add_dislike(request, recipe_id):
    if request.method == 'POST':
        recipe_dislikes = Dislikes.objects
        add_action(request, recipe_dislikes, recipe_id)
        return redirect('/' + str(recipe_id))


@login_required(login_url="/accounts/login")
def add_comment(request, recipe_id):
    if request.method == 'POST':
        recipe_comments = Comments.objects
        recipe_comments.create(user_id=request.user,
                               recipe_id=get_object_or_404(Recipe, pk=recipe_id),
                               comment=request.POST["comment"])

        return redirect('/' + str(recipe_id))


# Search engine by keyword
def search(request):
    if request.method == 'GET':
        keys = request.GET.get("search")
        page_number = request.GET.get("page")
        recipes = Recipe.objects.annotate(search=SearchVector('ingredients')).filter(search=SearchQuery(keys))
        paginator = Paginator(recipes, 36)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': keys})


# detail page for each recipe
def detail(request, recipe_id):
    recipe = get_object_or_404(Recipe, pk=recipe_id)
    summary = re.sub('<[^<]+?>', '', recipe.summary)
    steps = recipe.steps.split("|")
    if len(steps) < 2 and steps[0] == "":
        steps = []

    equipments = recipe.equipments.split(",")
    if len(equipments) < 2 and equipments[0] == "":
        equipments = []

    ingredients = recipe.ingredients.split(",")
    if len(ingredients) < 2 and ingredients[0] == "":
        ingredients = []

    dish_types = recipe.dish_types.split(",")
    if len(dish_types) < 2 and dish_types[0] == "":
        dish_types = []

    tags = []
    [tags.append(cuisine) for cuisine in recipe.cuisines.split(",")]
    if recipe.vegetarian == "True": tags.append("vegetarian")
    if recipe.vegan == "True": tags.append("vegan")
    if recipe.gluten_free == "True": tags.append("gluten free")
    if recipe.dairy_free == "True": tags.append("dairy free")
    if recipe.sustainable == "True": tags.append("sustainable")
    if recipe.very_healthy == "True": tags.append("very healthy")
    if recipe.very_popular == "True": tags.append("very popular")
    if recipe.low_fodmap == "True": tags.append("low fodmap")

    image = recipe.image
    if image is None:
        # default image
        pass

    # counts
    recipe_likes_count = Likes.objects.filter(recipe_id=recipe_id).count()
    recipe_dislikes_count = Dislikes.objects.filter(recipe_id=recipe_id).count()
    recipe_comments = Comments.objects.filter(recipe_id=recipe_id)

    # check weather this user has liked/disliked this recipe
    recipe_liked_by_user, recipe_disliked_by_user = False, False
    if Likes.objects.filter(user_id=request.user.id, recipe_id=recipe_id).exists():
        recipe_liked_by_user = True
    elif Dislikes.objects.filter(user_id=request.user.id, recipe_id=recipe_id).exists():
        recipe_disliked_by_user = True


    # comments analysis
    comments_with_sentiment = []
    if len(recipe_comments) > 0:
        sentiment_analyser = SentimentIntensityAnalyzer()
        pos_comments_count = 0
        neg_comments_count = 0
        sentiment_pct = 0.2

        for comment_obj in recipe_comments:
            result = sentiment_analyser.polarity_scores(comment_obj.comment)  # returns : {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            if result['pos'] > sentiment_pct:
                pos_comments_count += 1
                comments_with_sentiment.append((comment_obj, "pos"))
            elif result['neg'] > sentiment_pct:
                neg_comments_count += 1
                comments_with_sentiment.append((comment_obj, "neg"))
            else:
                comments_with_sentiment.append((comment_obj, ""))

        count = pos_comments_count + neg_comments_count

        if count > 0:
            pos_comments_pct = int((pos_comments_count / count) * 100)
            neg_comments_pct = 100 - pos_comments_pct
        else:
            pos_comments_pct, neg_comments_pct = 0, 0
    else:
        pos_comments_count = 0
        neg_comments_count = 0
        pos_comments_pct, neg_comments_pct = 0, 0

    # recommended recipes
    rec_recipes = content_based_rec(recipe_id)
    rec_recipes = Recipe.objects.filter(pk__in=rec_recipes)

    # get Twitter statistics
    stop_words = set(stopwords.words('english'))
    recipe_ingredients_keywords = [w for w in recipe.title.lower().split(" ") if w not in stop_words][:3]
    positive_tweets_pct, positive_tweets_count = get_users_feedbacks(keywords=recipe_ingredients_keywords,
                                                                     num_items=1000)

    return render(request, 'recipes/detail.html',
                  {'recipe': recipe, 'steps': steps, 'equipments': equipments, 'summary': summary,
                   'ingredients': ingredients, 'dish_types': dish_types, 'tags': tags,
                   'likes_count': recipe_likes_count, 'dislikes_count': recipe_dislikes_count,
                   'recipe_liked_by_user': recipe_liked_by_user, 'recipe_disliked_by_user': recipe_disliked_by_user,
                   'recommended_recipes': rec_recipes,
                   'positive_tweets_pct': positive_tweets_pct, 'positive_tweets_count':positive_tweets_count,
                   'pos_comments_pct': pos_comments_pct, 'neg_comments_pct': neg_comments_pct,
                   'pos_comments_count': pos_comments_count, 'neg_comments_count': neg_comments_count,
                   'comments_with_sentiment': comments_with_sentiment})


def recipe_generation_settings(request):

    return render(request, 'recipes/recipe_generation_settings.html', {"steps_beginnings_sentences": get_sentences_beginning()})


# generation recipe result
def recipe_generation(request):
    if request.method == 'POST':

        beginning_sentence = request.POST.get("beginning_sentence")
        steps_beginnings_sentences_dict = get_sentences_beginning()
        if beginning_sentence == "random":
            random_sentence = random.choice(list(steps_beginnings_sentences_dict.keys()))
            beginning_sentence = steps_beginnings_sentences_dict[random_sentence]
        else:
            beginning_sentence = steps_beginnings_sentences_dict[beginning_sentence]
        sentences_count = request.POST.get("sentences_count")
        generated_recipe = generate_sentences(generator_model, beginning_sentence, int(sentences_count))
        # get a list of similar recipes
        recipes_idx = get_similar_recipes(generated_recipe)
        similar_recipes = Recipe.objects.filter(pk__in=recipes_idx)

        # recipe keywords
        recipe_ingredients_keywords = [w for w in list(INGREDIENTS_KEYWORDS) if w in generated_recipe]

        return render(request, 'recipes/recipe_generation.html',
                      {'beginning_sentence': beginning_sentence, 'sentences_count': sentences_count,
                       'generated_recipe': generated_recipe,
                       'recipe_ingredients_keywords': recipe_ingredients_keywords,
                       'recommended_recipes': similar_recipes})
    else:
        return render(request, 'recipes/recipe_generation_settings.html')


# encode the generated recipe then find the similar ones
def get_similar_recipes(generated_recipe, num_results=12):

    stopwords = ['tbs', 'tablespoon', 'tablespoons', 'cup', 'tsp',
                 'tbsp', 'teaspoon', 'pound', 'tbsps', 'piece',
                 'ounce', 'slice', 'seed', 'medium', 'ozs',
                 'head', 'baby', 'inch', 'half', 'pinch', 'strip',
                 'handful', 'fat', 'stalk']
    nlp = spacy.load('en_core_web_lg')

    ing = str(generated_recipe).lower()
    ing = re.sub(",", " , ", ing)
    doc = nlp(ing)
    recipe_ings = [token.lemma_ for token in doc if not token.is_stop
                   and token.pos_ == 'NOUN'
                   and str(token) not in stopwords
                   and len(str(token)) > 2]

    # transform from text to array
    recipes_ing_df = pd.read_csv(RECIPES_INGREDIENTS_DF, sep='|', index_col=0)
    ing_cols = recipes_ing_df.columns
    encoded_recipe = [TOKEN_IN_TITLE_FACTOR if ing in recipe_ings else 0 for ing in ing_cols]
    recipe_array = np.array(encoded_recipe).reshape(1, -1)

    # get similar recipes through our kneighbor
    file = open(CLASSIFIER_NAME, 'rb')
    nearest_neighbors_algo = pickle.load(file)
    file.close()

    distances, nearest_idx = nearest_neighbors_algo.kneighbors(recipe_array, n_neighbors=num_results)
    real_idx_recipes = [recipes_ing_df.index[idx] for idx in nearest_idx[0]]

    return real_idx_recipes


# get recommended recipes based on its content
def content_based_rec(recipe_id, num_results=30):

    if not os.path.isfile(CLASSIFIER_NAME):
        print('Creating CLASSIFIER!')
        train_model()
    file = open(CLASSIFIER_NAME, 'rb')
    nearest_neighbors_algo = pickle.load(file)
    file.close()

    recipes_ing_df = pd.read_csv(RECIPES_INGREDIENTS_DF, sep='|', index_col=0)
    recipe_array = recipes_ing_df.loc[int(recipe_id), :].to_numpy().reshape(1, -1)

    distances, nearest_idx = nearest_neighbors_algo.kneighbors(recipe_array, n_neighbors=num_results)
    real_idx_recipes = [recipes_ing_df.index[idx] for idx in nearest_idx[0]]

    return real_idx_recipes


# train classifier model
# give a highest factor (factor of 10 instead of 1) to the ingredients included in the title
def train_model(max_ingredients=500):

    recipes = Recipe.objects.all()
    recipes_ids = list([recipe.id for recipe in recipes])
    recipes_ing = list(recipes.only('ingredients'))
    recipes_titles = list(recipes.only('title'))

    stopwords = ['tbs', 'tablespoon', 'tablespoons', 'cup', 'tsp',
                 'tbsp', 'teaspoon', 'pound', 'tbsps', 'piece',
                 'ounce', 'slice', 'seed', 'medium', 'ozs',
                 'head', 'baby', 'inch', 'half', 'pinch', 'strip',
                 'handful', 'fat', 'stalk']

    nlp = spacy.load('en_core_web_lg')

    all_ingredients = []
    ingredients_by_recipe = {}
    titles_by_recipe = {}
    st = time.time()
    # get most used keywords
    for recipe_id, ing_rec, title in zip(recipes_ids, recipes_ing, recipes_titles):
        ing = str(ing_rec).lower()
        # ing = re.sub("\b(tbs|\d+)\b", "", ing)
        ing = re.sub(",", " , ", ing)
        doc = nlp(ing)

        ing = [token.lemma_ for token in doc if not token.is_stop
               and token.pos_ == 'NOUN'
               and str(token) not in stopwords
               and len(str(token)) > 2]

        # remove duplicates
        ingredients_by_recipe[recipe_id] = list(set(ing))
        all_ingredients.append(ing)

        # titles formatting
        title = str(title).lower()
        title = re.sub(",", " , ", title)
        doc = nlp(title)

        ing_title = [token.lemma_ for token in doc if not token.is_stop
                     and token.pos_ == 'NOUN'
                     and str(token) not in stopwords
                     and len(str(token)) > 2]

        titles_by_recipe[recipe_id] = list(set(ing_title))
        all_ingredients.append(ing_title)

    # get most common ing
    all_ingredients = list(itertools.chain(*all_ingredients))
    ing_counts = Counter(all_ingredients).most_common(max_ingredients)
    all_ingredients = [i[0] for i in ing_counts]

    # create a matrix so that each column is a feature (ingredient)
    # and each line is a recipe vector
    recipes_ing_df = pd.DataFrame(columns=all_ingredients, index=recipes_ids)
    for idx in recipes_ing_df.index:
        recipe_ings = ingredients_by_recipe[idx]
        recipes_ing_df.loc[idx, :] = [1 if ing in recipe_ings else 0 for ing in all_ingredients]

        title_tokens = titles_by_recipe[idx]
        for token in title_tokens:
            if token in all_ingredients:
                recipes_ing_df.loc[idx, token] = TOKEN_IN_TITLE_FACTOR

    recipes_ing_df.to_csv(RECIPES_INGREDIENTS_DF, sep='|')

    recipes_ing_array = recipes_ing_df.to_numpy()
    nearest_neighbors_algo = NearestNeighbors(n_neighbors=1).fit(recipes_ing_array)
    file = open(CLASSIFIER_NAME, 'wb')
    pickle.dump(nearest_neighbors_algo, file)
    file.close()
    print("Training time: ", time.time() - st)
    print('CLASSFIER SAVED!!!')


def todays_special_recipe(user_id):

    if not os.path.isfile(CLASSIFIER_NAME):
        print('Creating CLASSIFIER!')
        train_model()

    # load models
    recipes_ing_df = pd.read_csv(RECIPES_INGREDIENTS_DF, sep='|', index_col=0)
    file = open(CLASSIFIER_NAME, 'rb')
    nearest_neighbors_algo = pickle.load(file)
    file.close()

    # pick a random recipe
    pks = Likes.objects.all().filter(user_id=user_id)
    liked_recipes_idx = list([recipe.recipe_id.id for recipe in pks])
    random_idx = random.choice(liked_recipes_idx)

    recipe_array = recipes_ing_df.loc[int(random_idx), :].to_numpy().reshape(1, -1)
    distances, nearest_idx = nearest_neighbors_algo.kneighbors(recipe_array, n_neighbors=10)
    real_idx_recipes = [recipes_ing_df.index[idx] for idx in nearest_idx[0]]

    # get for each similar recipe, its array
    similar_recipes = []
    for recipe_id in real_idx_recipes:
        recipe = list(recipes_ing_df.loc[int(recipe_id), :])
        recipe = [1 if x == TOKEN_IN_TITLE_FACTOR else x for x in recipe]
        similar_recipes.append(recipe)

    similar_recipes_arr = np.array(similar_recipes)
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(similar_recipes_arr)
    labels = kmeans.labels_
    recipes_per_cluster = {}
    for cluster in range(n_clusters):
        cluster_idx = [idx for i, idx in enumerate(real_idx_recipes) if labels[i] == cluster]
        recipes_per_cluster[cluster+1] = Recipe.objects.filter(pk__in=cluster_idx)

    return recipes_per_cluster