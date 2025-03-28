# Data Science Nanodegree Program by Udacity. 
    The dataset consists of pre-labeled tweets and messages from real-life disaster events. 
    The goal of this project is to build a Natural Language Processing (NLP) model capable of categorizing messages in real-time. 
    The project is divided into these core sections: 
    

## Getting Started Dependencies

    Python 3.11

    Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn, sqlite3, re, sqlite3
    NLP Libraries: NLTK
    SQLite Libraries: SQLAlchemy
    Model Serialization: Pickle
    Web App & Data Visualization: Flask, Plotly




## Quick Start
    
    Execute in following order (no code Change required)
        1. "Disaster_Response\ETL.py"
        2. "Disaster_Response\train_classifier.py" (will take 15-20min to train the model)
        3. "Disaster_Response\app\run.py"
        4. Go to http://0.0.0.0:3001/


    bash:
    cd
    ->
    python Disaster_Response/ETL.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    python Disaster_Response/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    python Disaster_Response/app/run.py
    http://0.0.0.0:3001/







# Data Cleaning

    The Cleaning is done in: 
    "Disaster_Response\ETL.py"
    for convenience ive already Prepared a cleaned File in the uploaded data
    if wanted it could be repeated anytime.
    


# Executing Program:

## Generating .plk

The Code is fully functional, unfortunately the *.pkl is too big to be uploaded,
to generate a new one (this takes a while) u have to execute following python program:

    Disaster_Response\train_classifier.py

## Change Path

U will need to open this and change the "db_path" so its working for ur system
once generated (estimated 20min) u will have the model



## Start Programm

Now run the following python script:
    
    Disaster_Response\app\run.py

once started u can test on this IP:

    Go to http://0.0.0.0:3001/
