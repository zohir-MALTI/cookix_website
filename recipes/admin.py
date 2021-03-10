from django.contrib import admin
from import_export.admin import ImportExportModelAdmin
from .models import Recipe


class RecipeAdmin(ImportExportModelAdmin):
    class Meta:
        model = Recipe

admin.site.register(Recipe, RecipeAdmin)
