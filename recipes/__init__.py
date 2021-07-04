#######################################
#######################################
########   My scripts import   ########
#######################################
#######################################

UTILS_FOLDER_PATH = "recipes/scripts/utils/"

from .scripts.recipe_generator import generate_sentences, generator_model, get_sentences_beginning, INGREDIENTS_KEYWORDS
from .scripts.twitter_topic_modelling import get_users_feedbacks


