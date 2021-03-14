from django.db import models

# percent_protein  percent_carbs percent_fat calories    saturated_fat_g carbohydrates_g
# sugar_g cholesterol_mg sodium_mg protein_g
# 	vitamin_B3_mg	selenium_microg	Phosphorus_mg	Iron_mg	vitamin_B2_mg	calcium_mg	vitamin_B1_mg
# folate_microg	potassium_mg	copper_mg	zinc_mg	Manganese_mg	Magnesium_mg	vitamin_B12_microg
# vitamin_B5_mg	vitamin_B6_mg	vitamin_E_mg Fiber_g	vitamin_A_IU	vitamin_D_microg	vitamin_K_microg
# vitamin_C_mg	alcohol_g	caffeine_g

class Recipe(models.Model):
    title = models.TextField()
    price_per_serving = models.FloatField()
    vegetarian = models.TextField()
    vegan = models.TextField()
    gluten_free = models.TextField()
    dairy_free = models.TextField()
    sustainable = models.TextField()
    very_healthy = models.TextField()
    very_popular = models.TextField()
    gaps = models.TextField()
    low_fodmap = models.TextField()
    ready_in_minutes = models.IntegerField()
    spoonacular_sourceUrl = models.URLField()
    image = models.TextField()
    aggregate_likes = models.IntegerField()
    spoonacular_score = models.IntegerField()
    health_score = models.IntegerField()
    dish_types = models.TextField()
    ingredients = models.TextField()
    cuisines = models.TextField()
    summary = models.TextField(blank=True)
    equipments = models.TextField(blank=True)
    steps = models.TextField(blank=True)
    users_likes = models.IntegerField(default=0)
    users_dislikes = models.IntegerField(default=0)

    def __str__(self):
        return self.title

    def summary_of_summary(self):
        return self.summary[:100]


