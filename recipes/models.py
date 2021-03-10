from django.db import models


class Recipe(models.Model):
    title = models.CharField(max_length=255)
    pricePerServing = models.CharField(max_length=255)
    weightPerServing = models.CharField(max_length=255)
    vegetarian = models.CharField(max_length=255)
    vegan = models.CharField(max_length=255)
    glutenFree = models.CharField(max_length=255)
    dairyFree = models.CharField(max_length=255)
    sustainable = models.CharField(max_length=255)
    veryHealthy = models.CharField(max_length=255)
    veryPopular = models.CharField(max_length=255)
    gaps = models.CharField(max_length=255)
    lowFodmap = models.CharField(max_length=255)
    ketogenic = models.CharField(max_length=255)
    whole30 = models.CharField(max_length=255)
    readyInMinutes = models.CharField(max_length=255)
    spoonacularSourceUrl = models.CharField(max_length=255)
    image = models.CharField(max_length=255)
    aggregateLikes = models.CharField(max_length=255)
    spoonacularScore = models.CharField(max_length=255)
    healthScore = models.CharField(max_length=255)
    percentProtein = models.CharField(max_length=255)
    percentFat = models.CharField(max_length=255)
    percentCarbs = models.CharField(max_length=255)
    dishTypes = models.CharField(max_length=255)
    ingredients = models.CharField(max_length=255)
    cuisines = models.CharField(max_length=255)
    calories = models.CharField(max_length=255)
    """
    Fat / g = models.CharField(max_length=255)
    Saturated
    Fat / g = models.CharField(max_length=255)
    Carbohydrates / g = models.CharField(max_length=255)
    Sugar / g = models.CharField(max_length=255)
    Cholesterol / mg = models.CharField(max_length=255)
    Sodium / mg = models.CharField(max_length=255)
    Protein / g = models.CharField(max_length=255)
    Vitamin
    B3 / mg = models.CharField(max_length=255)
    Selenium / µg = models.CharField(max_length=255)
    Phosphorus / mg = models.CharField(max_length=255)
    Iron / mg = models.CharField(max_length=255)
    Vitamin
    B2 / mg = models.CharField(max_length=255)
    Calcium / mg = models.CharField(max_length=255)
    Vitamin
    B1 / mg = models.CharField(max_length=255)
    Folate / µg = models.CharField(max_length=255)
    Potassium / mg = models.CharField(max_length=255)
    Copper / mg = models.CharField(max_length=255)
    Zinc / mg = models.CharField(max_length=255)
    Manganese / mg = models.CharField(max_length=255)
    Magnesium / mg = models.CharField(max_length=255)
    Vitamin
    B12 / µg = models.CharField(max_length=255)
    Vitamin
    B5 / mg = models.CharField(max_length=255)
    Vitamin
    B6 / mg = models.CharField(max_length=255)
    Vitamin
    E / mg = models.CharField(max_length=255)
    Fiber / g = models.CharField(max_length=255)
    Vitamin
    A / IU = models.CharField(max_length=255)
    Vitamin
    D / µg = models.CharField(max_length=255)
    Vitamin
    K / µg = models.CharField(max_length=255)
    Vitamin
    C / mg = models.CharField(max_length=255)
    Alcohol / g = models.CharField(max_length=255)
    Caffeine / g = models.CharField(max_length=255)
    """

