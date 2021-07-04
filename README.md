# Cookix

The website is available  [at this adress](http://157.230.24.228:8000/) ! 

To run the website on your local machine, here are the steps:
* After creating a python environment, you can install some the required libraries:

		pip install django==3.0 django-import-export tensorflow==2.4.0 psycopg2-binary scikit-learn==0.24.0 spacy==2.3.5 nltk==3.5 tweepy pandas
		python -m spacy download en_core_web_lg

* Make migrations if needed:

		python manage.py makemigrations
		python manage.py migrate

* Run the server:

		python manage.py runserver

* Access to http://localhost:8000/
