import sys
import sqlalchemy
from sqlalchemy import create_engine

import pandas as pd
import numpy as np


def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv file.
    
    INPUT:
    messages_filepath - (str) file path for messages
    categories_filepath - (str) file path for categories
    
    OUTPUT:
    None
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    
    return df

def _encode_categories(df):
    '''
    Apply one-hot encoding for categories.
    
    INPUT:
    df - (pandas dataframe) 
        dataframe which contains categories
    
    OUTPUT:
    df - (pandas dataframe) 
        changed df
    '''
    
    categories_dummy = df.categories.str.split(";", expand=True)
    first_row = categories_dummy.iloc[0, :]
    category_colnames = first_row.apply(lambda x: x.split("-")[0])
    categories_dummy.columns = category_colnames
    
    for column in categories_dummy:
        categories_dummy[column] = categories_dummy[column].apply(lambda x: x.split('-')[-1])
        categories_dummy[column] = categories_dummy[column].astype(int)
    
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories_dummy], axis=1)
    
    df = df[df.related != 2]
    
    return df

def _remove_duplicates(df):
    '''
    Remove duplicated data from dataframe. 
    There are two types of duplicated data: 
    1. Redundant data, which content are exactly the same. 
    2. Conflict data, which id relates to different categories. 
    
    INPUT:
    df - (pandas dataframe)
        dataframe for removing duplicates
    OUTPUT:
    df - (pandas dataframe) 
        changed df
    '''
    df.drop_duplicates(inplace=True)
    df.drop_duplicates('id', keep=False, inplace=True)
    
    return df


def clean_data(df):
    '''
    Clean input dataframe.
    
    INPUT:
    df - (pandas dataframe)
        dataframe to be cleaned
        
    OUTPUT:
    df - (pandas dataframe)
        changed df
    '''
    df = _encode_categories(df)
    df = _remove_duplicates(df)
    
    return df


def save_data(df, database_filename):
    '''
    Save dataframe into database
    
    INPUT:
    df - (pandas dataframe)
        dataframe to be saved
    OUTPUT:
    database_filename - (str)
        path for database 
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessagesVsCategories', engine, index=False)  


def main():
    '''
    Messages and categorie data ETL
    
    INPUT:
    argv[1] - (str)
        path for messages file
    argv[2] - (str)
        path for categories file
    argv[3] - (str)
        path for database
        
    OUTPUT:
    None
    '''
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