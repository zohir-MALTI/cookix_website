import time
import numpy as np
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required

from . import TwitterStats
from .models import *
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
# from cookix_website import twitter_stats


CLASSIFIER_NAME = "recommendation_classifier.pickle"
RECIPES_INGREDIENTS_DF = "recipes_by_ing_df.csv"
TOKEN_IN_TITLE_FACTOR = 10

def home(request):
    recipes = Recipe.objects.all()[4000:4009]
    recipes_min = recipes[:100]
    # print(len(recipes))
    user_id = request.user.id
    if user_id is not None:
        clusters = custom_recipes_clusters(user_id)
    return render(request, 'recipes/home.html', {'recipes': recipes_min,
                                                 'recipes_count': Recipe.objects.count(),
                                                 'todays_special': recipes_min[0],
                                                 'clusters': clusters})


def custom_recipes_clusters(user_id, n_clusters=6, num_results=30):

    if not os.path.isfile(CLASSIFIER_NAME):
        print('Creating CLASSIFIER!')
        train_model()

    recipes_ing_df = pd.read_csv(RECIPES_INGREDIENTS_DF, sep='|', index_col=0)
    file = open(CLASSIFIER_NAME, 'rb')
    nearest_neighbors_algo = pickle.load(file)
    file.close()

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

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(similar_recipes_arr)
    #labels = zip(similar_recipes_idx, kmeans.labels_)
    labels = kmeans.labels_
    recipes_per_cluster = {}
    for cluster in range(n_clusters):
        cluster_idx = [idx for i, idx in enumerate(similar_recipes_idx) if labels[i] == cluster]
        recipes_per_cluster[cluster+1] = Recipe.objects.filter(pk__in=cluster_idx)


    return recipes_per_cluster



@login_required(login_url="/accounts/login")
def favorites(request):
    pks = Likes.objects.all().filter(user_id=request.user.id)
    pks = [recipe.recipe_id for recipe in pks]
    liked_recipes = Recipe.objects.filter(title__in=list(pks)).all()
    # liked_recipes = Recipe.objects.filter(pk__in=Likes.objects.filter(user_id=request.user.id))

    return render(request, 'recipes/favorites.html', {'recipes': liked_recipes})


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


def search(request):
    if request.method == 'GET':
        keys = request.GET.get("search")
        page_number = request.GET.get("page")
        recipes = Recipe.objects.filter(
            Q(title__icontains=keys) #| Q(ingredients__icontains=keys)
            #Q(equipments__icontains=keys)
        )  # filter(search='cheese')  / filter(body_text__search='cheese')
        paginator = Paginator(recipes, 24)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': keys})


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

    # recommended_recipes = Recipe.objects.all()
    # recommended_recipes = recommended_recipes[2800:2803]

    recipe_likes_count = Likes.objects.filter(recipe_id=recipe_id).count()
    recipe_dislikes_count = Dislikes.objects.filter(recipe_id=recipe_id).count()
    # print("cooo: ", recipe_likes_count)

    # recommended recipes
    rec_recipes = content_based_rec(recipe_id)
    rec_recipes = Recipe.objects.filter(pk__in=rec_recipes)

    # get Twitter statistics
    # tw = TwitterStats().get_tweets(["couscous", "salmon", "Shrimp"])
    likes_pct, hates_pct = TwitterStats().analyze("couscous OR salmon OR Shrimp")


    return render(request, 'recipes/detail.html',
                  {'recipe': recipe, 'steps': steps, 'equipments': equipments, 'summary': summary,
                   'ingredients': ingredients, 'dish_types': dish_types, 'tags': tags,
                   'likes_count': recipe_likes_count,
                   'dislikes_count': recipe_dislikes_count,
                   'recommended_recipes': rec_recipes,
                   'likes_pct': likes_pct, 'hates_pct': hates_pct})


# give a highest factor (factor of 10 instead of 1) to the ingredients included in the title
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


def train_model(max_ingredients=500):

    recipes = Recipe.objects.all()
    recipes_ids = list([recipe.id for recipe in recipes])#[:n]
    recipes_ing = list(recipes.only('ingredients'))#[:n]
    recipes_titles = list(recipes.only('title'))#[:n]

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

    print("Training time: ",time.time() - st)
    all_ingredients = list(itertools.chain(*all_ingredients))
    ing_counts = Counter(all_ingredients).most_common(max_ingredients)
    all_ingredients = [i[0] for i in ing_counts]

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
    print('CLASSFIER SAVED!!!')

    # sig_results = pd.DataFrame(sigmoid_kernel(recipes_ing_df, recipes_ing_df),
    #                            index=recipes_ids,
    #                            columns=recipes_ids)
    #
    # sig_results.to_csv('sigmoid_kernel_ingredients.csv', sep='|')


