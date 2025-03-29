# Data Science Nanodegree Program by Udacity. 
The dataset consists of pre-labeled tweets and messages from real-life disaster events. 
The goal of this project is to build a Natural Language Processing (NLP) model capable of categorizing messages in real-time. 
The project is divided into these core sections: 

## Project Overview

This project focuses on building a Natural Language Processing (NLP) model 
designed to categorize messages from real-life disaster events, such as natural disasters or crises. 
The model processes pre-labeled tweets and messages in real-time, enabling timely and accurate categorization of disaster-related information.
Practical Impact & Benefits

In the event of a disaster, this application can provide immediate value by quickly sorting and identifying critical messages. 
By automating the categorization process, it ensures that emergency responders, organizations, and even affected individuals can access essential information quickly. 
This can help direct resources to where they are needed most, prioritize emergency responses, and inform people of safety measures or updates in real-time.

For disaster management organizations, this model could improve their situational awareness and decision-making by providing a clear understanding of public sentiment 
and urgent requests in the midst of a crisis. It can also support non-profit organizations and governmental agencies in coordinating aid and optimizing rescue efforts.

Through faster response times and better organization of critical data, the application can significantly reduce confusion during high-stress situations, 
ultimately saving lives and improving recovery efforts in disaster scenarios.
    

## Getting Started Dependencies

    Python 3.11

    Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn, sqlite3, re, sqlite3
    NLP Libraries: NLTK
    SQLite Libraries: SQLAlchemy
    Model Serialization: Pickle
    Web App & Data Visualization: Flask, Plotly




## Quick Start
    bash:
    1. cd (to the folder "Disaster_Response")
    
    ->
    Important! you have to generate the .plk file (Step 3.)  which will take place in the models subfolder (placeholder,txt is just there to keep the folder)
    i cannot upload it since its a big file -> this process will take 15-20minutes

    
    2. python ETL.py data/disaster_messages.csv data/disaster_categories.csv data/Udacity_disaster.db
    3. python train_classifier.py data/Udacity_disaster.db models/classifier.pkl
    4. python app/run_db.py
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
