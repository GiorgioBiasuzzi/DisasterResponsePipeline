#import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    This function merges messages and categories and returns only one of the binary values (0,1) for each of the columns within the categories dataframe
    -Drop original categories columns
    -Concatenate the df with the new categories created in the load_data function
    Args:   messages_filepath: path where the messages dataframe is stored
            categories_filepath: path where the categories dataframe is stored
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how='left', on='id')
    # split categories dataset
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        ### set each value to be the last character of the string
        # convert the Series to be of type string
        categories[column] = categories[column].astype(str)
        # keep only the last character of each string (the 1 or 0)
        categories[column] = categories[column].str.strip().str[-1]
   
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df
    pass


def clean_data(df):
    """
    This function cleans the dataframe resulted from the merge between messages and categories (df)
    -Drop the duplicates
    Args: df dataframe passed from load_data function
    """
    # drop duplicates
    df = df.drop_duplicates(subset='id', keep='first')
    
    return df

    pass


def save_data(df, database_filename):
    """
    This function saves the dataframe into the database
    Args:   df: final dataframe
            database_filename: path where the database is stored
    """
    engine = create_engine(database_filename)
    df.to_sql('TblHelpRequests', engine, index=False, if_exists = 'replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()