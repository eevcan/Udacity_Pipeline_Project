import json
import plotly
import pandas as pd
import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)
nltk.download('punkt')
nltk.download('wordnet')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Get the absolute path for the database and model files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Get the root directory of the project
data_dir = os.path.join(project_root, 'data')  # Data directory
model_dir = os.path.join(project_root, 'models')  # Models directory

# Update paths for database and model
database_path = os.path.join(data_dir, 'Udacity_disaster.db')
engine = create_engine(f'sqlite:///{database_path}')

# Load data
df = pd.read_sql_table('Udacity_disaster_table', engine)

# Load model
model = joblib.load(os.path.join(model_dir, 'classifier.pkl'))


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html page with the results
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
