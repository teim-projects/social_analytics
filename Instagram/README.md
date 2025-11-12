To run the project:

Note: Before running any command check if you are in path 
"...\Social_media_monitoring_platform\Instagram>" i.e. inside the Instagram folder

in the .env files inside Instagram update DATA_PATH and MODEL_PATH as per your system

To install all requirments run the following command:
pip install -r requirments.txt

To run the project run the following command:
python manage.py run-all

then open http://127.0.0.1:8000/ to view the project output

------------------------------------------------------------------------------------------------------------

If you want to run individual dashboards and then main project then run the following commands in separate terminals:

python -m streamlit run streamlit-app/dashboard.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8502
python -m streamlit run streamlit-app/engagement_rate.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8503
python -m streamlit run streamlit-app/top_posts.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8504
python -m streamlit run streamlit-app\instagram_engagement.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8505

then finally 

python manage.py runserver

------------------------------------------------------------------------------------------------------------

If facing MySQL error:
check settings.py if it has DATABASES as
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

then in a terminal run the following:
python manage.py migrate
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 
