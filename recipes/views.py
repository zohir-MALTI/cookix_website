from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Recipe

@login_required
def home(request):
    recipes = Recipe.objects.all()
    # print(len(recipes))
    recipes = recipes[:3]
    # print(len(recipes))
    return render(request, 'recipes/home.html', {'recipes': recipes})


def detail(request, recipe_id):
    recipe = get_object_or_404(Recipe, pk=recipe_id)
    steps = recipe.steps.split("|")
    if len(steps)<2 and steps[0]=="":
        steps = []
    return render(request, 'recipes/detail.html', {'recipe': recipe, 'steps': steps})


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
