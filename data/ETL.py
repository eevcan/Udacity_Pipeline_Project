#!/usr/bin/env python
# coding: utf-8

# # 1. Load datasets and import libraries.
# 
#     Load the messages.csv into dataframe and check first few rows.
#     Import Python libraries.
#     Load categories.csv into dataframe and check first few rows.
# 

# In[ ]:





# In[15]:


import pandas as pd
from sqlalchemy import create_engine
import os


# In[16]:


message = pd.read_csv(r'C:\Users\eev\Documents\Udacity\Disaster_Response\data\disaster_messages.csv')
message.head(10)  # Show the first 10 rows


# In[17]:


categories = pd.read_csv(r'C:\Users\eev\Documents\Udacity\Disaster_Response\data\disaster_categories.csv')
categories.head(10)


# In[18]:


print('Row Number and colum number are: {} and {}'.format(message.shape[0],message.shape[1]))
print('Row Number and colum number are: {} and {}'.format(categories.shape[0],categories.shape[1]))


# # 2. Merging datasets.
# 
#     Combine the messages and categories datasets using the common id.
#     Save this combined dataset as df, which will be cleaned later.
# 

# In[19]:


df = message.merge(categories, on='id')
df.head(10) 


# In[20]:


print('Row Number and colum number are: {} and {}'.format(df.shape[0],df.shape[1]))


# # 3. Split categories into different category columns.
# 
#     Split the values in categories column by ';' so each value is a separate column. This method will be useful! Don't forget to use expand=True.
#     Create column names for the categories data from the first row of categories dataframe.
#     Rename columns of categories with the new names.

# In[21]:


# Create a dataframe for the categories and pull them apart
categories = df['categories'].str.split(pat=';', expand=True)
categories.head(5)  # Display first 5 rows


# In[22]:


#extract the information
row = categories.iloc[[1]]
category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
print(category_colnames)


# In[23]:


'''
This Column names are not telling me anything, if i give this to someone else
he propably has to analyze it from the beginning again
'''
categories.columns = category_colnames
categories.head()


# # 4. Convert category values to 0/1
# 
#     Go through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Change the string to a number.
#     You can use normal string actions on Pandas Series, like indexing, by adding .str after the Series. You might need to first change the Series to string type, which you can do with astype(str).
# 

# In[24]:


for column in categories:
    # Set each value to the last character of the string
    categories[column] = categories[column].astype(str).str[-1:]

    # Convert the column from string to numeric (integer)
    categories[column] = pd.to_numeric(categories[column], errors='coerce')  # 'coerce' will set errors to NaN

    # Replace NaN values with 0 if there are any (in case of invalid data)
    categories[column] = categories[column].fillna(0).astype(int)


# # 5. Replace categories column in df with the new category columns.
# 
#     Drop the categories column from the df dataframe because it's not needed anymore.
#     Combine the df and categories dataframes.

# In[25]:


# Remove the original categories column from `df`
df.drop(['categories'], axis=1, inplace=True)
df.head(5)


# In[26]:


# concat the original df with the new categories dataframe
df = pd.concat([df,categories], join='inner', axis=1)
df.head(10)


# # 6. Remove duplicates.
# 
#     Check how many duplicates are in the dataset.
#     Remove the duplicates.
#     Confirm that the duplicates were removed.

# In[27]:


# Check how many duplicates are in the dataset before removing them
print('before: {}'.format(sum(df.duplicated())))
df.drop_duplicates(inplace=True)  # Remove duplicates
print('after: {}'.format(sum(df.duplicated())))


# # 7. Save the clean dataset into an sqlite database.
# 
#     You can use pandas to_sql method with the SQLAlchemy library. Don’t forget to import create_engine from SQLAlchemy in the first cell of the notebook to use it below.
# 

# In[29]:


database_filepath = r'C:\Users\eev\Documents\Udacity\Disaster_Response\data\Udacity_disaster.db'
engine = create_engine('sqlite:///' + database_filepath)  # Create a connection to the SQLite database
table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"  # Get table name from the file path
df.to_sql(table_name, engine, index=False, if_exists='replace')  # Save the dataframe to the database


# In[ ]:




