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
nltk.download('punkt')  # Download punkt tokenizer
nltk.download('wordnet')  # Download WordNet Lemmatizer

def tokenize(text):
    """Tokenizes and lemmatizes input text"""
    tokens = word_tokenize(text)  # Tokenize the text into words
    lemmatizer = WordNetLemmatizer()  # Create lemmatizer object

    # Lemmatize each token, convert to lowercase and strip whitespace
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Get the absolute path for the database and model files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Get the root directory of the project
data_dir = os.path.join(project_root, 'data')  # Data directory
model_dir = os.path.join(project_root, 'models')  # Models directory

# Update paths for database and model
database_path = os.path.join(data_dir, 'Udacity_disaster.db')
engine = create_engine(f'sqlite:///{database_path}')  # Create SQLAlchemy engine for the database

# Load data
df = pd.read_sql_table('Udacity_disaster_table', engine)  # Read the table into a DataFrame

# Load model
model = joblib.load(os.path.join(model_dir, 'classifier.pkl'))  # Load the trained model using joblib


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Renders the index page with visuals of message genre and category distributions"""
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']  # Count messages by genre
    genre_names = list(genre_counts.index)

    # Create visuals for genre distribution
    genre_graph = {
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

    # Extract category counts (from the columns starting from index 4 onward)
    category_counts = df.iloc[:, 4:].sum()  # Sum values for each category
    category_names = list(category_counts.index)

    # Create visuals for category distribution
    category_graph = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }

    # Combine both graphs into a list
    graphs = [genre_graph, category_graph]
    
    # Encode plotly graphs in JSON format for rendering
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """Handles user query, predicts categories, and displays classification results"""
    query = request.args.get('query', '')  # Get the user input query

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]  # Predict the classification labels for the query
    classification_results = dict(zip(df.columns[4:], classification_labels))  # Map predictions to category names

    # Extract data for visuals (genre and category counts)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Create the first bar chart (Genre Distribution)
    genre_graph = {
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

    # Extract category counts (from the columns starting from index 4 onward)
    category_counts = df.iloc[:, 4:].sum()
    category_names = list(category_counts.index)

    # Create the second bar chart (Category Distribution)
    category_graph = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],

        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }

    # Combine both graphs into a list
    graphs = [genre_graph, category_graph]
    
    # Encode plotly graphs in JSON format
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html page with the results and the graphs
    return render_template('go.html', query=query, classification_result=classification_results, graphJSON=graphJSON, ids=ids)


def main():
    """Runs the Flask app"""
    app.run(host='0.0.0.0', port=3001, debug=True)  # Start the Flask application on port 3001

if __name__ == '__main__':
    main()  # Execute the main function when the script is run
