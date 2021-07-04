from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('<int:recipe_id>', views.detail, name='detail'),
    path('<int:recipe_id>/add_like', views.add_like, name='add_like'),
    path('<int:recipe_id>/add_dislike', views.add_dislike, name='add_dislike'),
    path('<int:recipe_id>/add_comment', views.add_comment, name='add_comment'),
    path('search', views.search, name='search'),
    path('favorites', views.favorites, name='favorites'),
    path('recipe-generation-settings', views.recipe_generation_settings, name='recipe_generation_settings'),
    path('recipe-generation', views.recipe_generation, name='recipe_generation'),
    path('dish_types', views.dish_types, name='dish_types'),
    path('diets/<slug:diet_type>', views.diet_types, name='diet_types'),
    path('dishes/<slug:dish_type>', views.dish_types, name='dish_types'),
    path('cuisines/<slug:cuisine_type>', views.cuisine_types, name='cuisine_types'),
    path('about-us', views.about_us, name='about_us'),
    # path('recommend', views.recommend_recipes, name='recommend'),
]


