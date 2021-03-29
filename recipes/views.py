import re
from django.contrib.postgres.search import SearchQuery, SearchVector
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Recipe


def home(request):
    recipes = Recipe.objects.all()[4000:4009]
    print("aaaaa",recipes)
    print("type ",type(recipes))
    print("len ",len(recipes))
    recipes_count = len(recipes)
    recipes_min = recipes[:10]
    # print(len(recipes))
    # print(len(recipes))
    return render(request, 'recipes/home.html', {'recipes': recipes_min,
                                                 'recipes_count': recipes_count,
                                                 'todays_special': recipes_min[0]})


def favorites(request):
    recipes = Recipe.objects.all()
    recipes = recipes[4000:4006]
    return render(request, 'recipes/home.html', {'recipes': recipes})


def detail(request, recipe_id):
    recipe = get_object_or_404(Recipe, pk=recipe_id)
    summary = re.sub('<[^<]+?>', '', recipe.summary)
    steps = recipe.steps.split("|")
    if len(steps)<2 and steps[0] == "":
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
    if recipe.vegetarian == "True" : tags.append("vegetarian")
    if recipe.vegan == "True" : tags.append("vegan")
    if recipe.gluten_free == "True" : tags.append("gluten free")
    if recipe.dairy_free == "True" : tags.append("dairy free")
    if recipe.sustainable == "True" : tags.append("sustainable")
    if recipe.very_healthy == "True" : tags.append("very healthy")
    if recipe.very_popular == "True" : tags.append("very popular")
    if recipe.low_fodmap == "True" : tags.append("low fodmap")

    image = recipe.image
    if image is None:
        # default image
        pass

    recommended_recipes = Recipe.objects.all()
    recommended_recipes = recommended_recipes[2800:2803]

    return render(request, 'recipes/detail.html', {'recipe': recipe, 'steps': steps, 'equipments': equipments, 'summary': summary,
                                                   'ingredients': ingredients, 'dish_types': dish_types, 'tags': tags,
                                                   'recommended_recipes': recommended_recipes})

@login_required(login_url="/accounts/signup")
def add_like(request, recipe_id):
    if request.method == 'POST':
        recipe = get_object_or_404(Recipe, pk=recipe_id)
        recipe.users_likes += 1
        recipe.save()
        return redirect('/'+str(recipe.id))


@login_required(login_url="/accounts/signup")
def add_dislike(request, recipe_id):
    if request.method == 'POST':
        recipe = get_object_or_404(Recipe, pk=recipe_id)
        recipe.users_dislikes += 1
        recipe.save()
        return redirect('/'+str(recipe.id))


def search(request):
    if request.method == 'GET':
        keys = request.GET.get("search")
        print("keys:" ,keys)
        page_number = request.GET.get("page")
        recipes = Recipe.objects.all()\
            .annotate(search = SearchVector('title', 'ingredients', 'summary', 'cuisines', 'equipments'))\
            .filter(title__icontains=keys)  # filter(search='cheese')  / filter(body_text__search='cheese')
        paginator = Paginator(recipes, 3)
        try:
            recipes_obj = paginator.get_page(page_number)
        except PageNotAnInteger:
            recipes_obj = paginator.get_page(1)
        except EmptyPage:
            recipes_obj = paginator.get_page(1)

        recipes2 = SearchQuery(keys, search_type='raw')

        print("recipes_obj:" ,recipes_obj)
        print("recipes2:" ,recipes2)
        # print("count:" ,Recipe.recipes_count())
        # return redirect('/'+str(recipe.id))
        return render(request, 'recipes/search.html', {'recipes': recipes_obj, 'result_count': len(recipes),
                                                       'query': keys})
