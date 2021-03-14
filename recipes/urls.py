from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('<int:recipe_id>', views.detail, name='detail'),
    path('<int:recipe_id>/add_like', views.add_like, name='add_like'),
    path('<int:recipe_id>/add_dislike', views.add_dislike, name='add_dislike'),
]
