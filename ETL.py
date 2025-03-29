import pandas as pd
import os
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load the messages and categories data from CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the datasets on 'id'
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    # Create a dataframe for the categories and split them
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Extract category names from the first row and set them as column names
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    # Clean the categories columns by setting each value to the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = pd.to_numeric(categories[column], errors='coerce')  # Convert to numeric
        categories[column] = categories[column].fillna(0).astype(int)  # Replace NaNs with 0

    # Remove the original 'categories' column from the dataframe and concatenate the cleaned categories
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    # Create a connection to the SQLite database
    engine = create_engine('sqlite:///' + database_filename)
    table_name = os.path.basename(database_filename).replace(".db", "") + "_table"
    
    # Save the dataframe to the database
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    # Define the file paths directly in the code
    #messages_filepath = 'data/disaster_messages.csv'
    #categories_filepath = 'data/disaster_categories.csv'
    #database_filepath = 'data/DisasterResponse.db'
    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    # Load data
    df = load_data(messages_filepath, categories_filepath)

    # Clean data
    df = clean_data(df)

    # Save cleaned data to the database
    save_data(df, database_filepath)

if __name__ == '__main__':
    main()
