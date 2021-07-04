from django.contrib import admin
from django.urls import path, include
# from cookix_website.recipes import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('recipes.urls')),
    path('accounts/', include('accounts.urls')),
]
