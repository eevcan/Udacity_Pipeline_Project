import sqlite3
import pandas as pd
import re
import pickle
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_table_name(db_path):
    """Fetches the table name from the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return tables[0][0] if tables else None  # Assuming first table is the right one

def load_data(db_path):
    """Loads data from the SQLite database"""
    table_name = get_table_name(db_path)
    if not table_name:
        raise ValueError("No table found in the database.")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    
    X = df['message']  # Features (text messages)
    y = df.iloc[:, 4:]  # Targets (categories, adjust column indices if needed)
    return X, y

def tokenize(text):
    """Tokenizes and lemmatizes text data"""
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Remove non-alphabetic characters and lowercase the text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # Tokenize the text
    tokens = text.split()
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized_tokens

def build_pipeline():
    """Creates a machine learning pipeline"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Use MultiOutputClassifier
    ])
    
    # Define parameter grid for tuning
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [None, 10]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py database_filepath model_filepath")
        sys.exit(1)

    db_path = sys.argv[1]   # Get database path from command line
    model_path = sys.argv[2]  # Get model save path from command line

    print("Loading data...")
    X, y = load_data(db_path)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building model...")
    model = build_pipeline()
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Evaluate each category (column) separately
    for i, column in enumerate(y_test.columns):
        print(f"Evaluating category: {column}")
        print(classification_report(y_test[column], y_pred[:, i]))
    
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
