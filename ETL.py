import pandas as pd
import os
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads and merges messages and categories data from CSV files"""
    # Load the messages and categories data from CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the datasets on 'id' column
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """Cleans the merged dataset by splitting categories and converting values to numeric"""
    # Create a dataframe for the categories and split them
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Extract category names from the first row and set them as column names
    row = categories.iloc[[1]]  # Get the second row for category names (index starts at 0)
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]  # Get category names without the '-0' part
    categories.columns = category_colnames  # Set category names as column headers
    
    # Clean the categories columns by setting each value to the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]  # Extract last character (0 or 1 or 2)
        categories[column] = pd.to_numeric(categories[column], errors='coerce')  # Convert to numeric values
        categories[column] = categories[column].fillna(0).astype(int)  # Replace NaN values with 0 and convert to integers
    
    # Drop rows where any category has a value of 2
    df = df[~(categories == 2).any(axis=1)]  # Remove rows where any column in categories has the value 2

    # Remove the original 'categories' column from the dataframe and concatenate the cleaned categories
    df.drop(['categories'], axis=1, inplace=True)  # Drop the original 'categories' column
    df = pd.concat([df, categories], join='inner', axis=1)  # Add cleaned categories as new columns
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """Saves the cleaned dataframe to an SQLite database"""
    # Create a connection to the SQLite database
    engine = create_engine('sqlite:///' + database_filename)  # Connect to SQLite database
    table_name = os.path.basename(database_filename).replace(".db", "") + "_table"  # Set table name based on database filename
    
    # Save the dataframe to the database
    df.to_sql(table_name, engine, index=False, if_exists='replace')  # Save to the database, replace if table exists

def main():
    """Main function to load, clean, and save disaster response data"""
    # Define the file paths directly in the code
    #messages_filepath = 'data/disaster_messages.csv'
    #categories_filepath = 'data/disaster_categories.csv'
    #database_filepath = 'data/DisasterResponse.db'
    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]  # Get file paths from command line arguments

    # Load data
    df = load_data(messages_filepath, categories_filepath)

    # Clean data
    df = clean_data(df)

    # Save cleaned data to the database
    save_data(df, database_filepath)

if __name__ == '__main__':
    main()
