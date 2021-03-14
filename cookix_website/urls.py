from django.contrib import admin
from django.urls import path, include
# from cookix_website.recipes import views


urlpatterns = [
    path('admin/', admin.site.urls),
    # path('', views.home, name='home'),
    path('', include('recipes.urls')),
    path('accounts/', include('accounts.urls')),
    # path('recipes/', include('recipes.urls')),
]
