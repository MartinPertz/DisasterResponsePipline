# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine  



def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets
    
    INPUT:
    messages_filepath: Filepath for csv file containing messages dataset 
    categories_filepath: Filepath for csv file containing categories dataset 
       
    OUTPUT:
    df: Dataframe with merged messages and categories datasets.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, left_on = 'id', right_on = 'id', how = 'inner')

    return df



def clean_data(df):
    """
    Cleans the data
    
    INPUT:
    df dataset 
    
    OUTPUT: 
    Clean dataframe, no duplicates
    """

    # create a dataframe of the 36 individual category columns
    categories_new = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    first_row_category = categories_new.iloc[0]

    category_colnames = first_row_category.apply(lambda x: x[: -2])

    # rename the columns of `categories`
    categories_new.columns = category_colnames

    categories_new.astype(str)
    
    for col in categories_new:
        
        # set each value to be the last character of the string
        categories_new[col] = categories_new[col].str[-1] 
    
        # convert column from string to numeric
        categories_new[col] = pd.to_numeric(categories_new[col])

    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_new], axis = 1, sort = False)

    # "child_alone" column has only one value "0", remove "child_alone"''
    df = df.drop('child_alone', axis = 1)

    # "related" column has the values “0”, “1”, “2”,  remove data with value "2" in "related" column
    df = df[df.related != 2]
    
    return df


def save_data(df, database_filename):
    """
    Save cleaned data
    
    INPUT:
    df: cleaned dataframe
    database_filename
       
    OUTPUT:
    None
    """
    engine = create_engine('sqlite:///' +database_filename)
    df.to_sql('DisasterMessages', engine, index=False)

    


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