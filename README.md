**
This project is a part of the Data Science Nanodegree Program by Udacity. The dataset consists of pre-labeled tweets and messages from real-life disaster events. 
The goal of this project is to build a Natural Language Processing (NLP) model capable of categorizing messages in real-time.
The project is divided into these core sections:**


    Data Processing: Building an ETL pipeline to extract data from the source, clean it, and store it in a SQLite database.
    Machine Learning Pipeline: Creating a pipeline to train a model that can classify text messages into various categories.
    Web Application: Deploying a web app to display model results in real-time.

**
Getting Started
Dependencies**

    Python 3.11

    Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn, sqlite3
    NLP Libraries: NLTK
    SQLite Libraries: SQLAlchemy
    Model Serialization: Pickle
    Web App & Data Visualization: Flask, Plotly

Executing Program:

    You can run the following commands in the project's directory to set up the database, train model and save the model.
        To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
        To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

    Run the following command in the app's directory to run your web app. python run.py

    Go to http://0.0.0.0:3001/
