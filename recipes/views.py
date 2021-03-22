import re

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Recipe


def home(request):
    recipes = Recipe.objects.all()
    # print(len(recipes))
    recipes = recipes[4000:4006]
    # print(len(recipes))
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


