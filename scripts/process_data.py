import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np






def load_data(messages_filepath, categories_filepath):
    '''
    Function for loading the 2 datasets into 1 master dataset
    Args:   messages_filepath: The file path to the messages.csv file
    categories_filepath: The file path to the categories.csv file
    Returns: A pandas DataFrame containing both files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = ['id'])
    
    print(df.shape)
    return df


def clean_data(df):
    '''
    Cleaning function: drops duplicates, changes caegories to numerical
    Args: Pandas df
    Returns: df.drop_duplicates()
    '''
    categories = df['categories'].str.split(';',expand=True)
    firstrow = categories.iloc[0,:]
    category_colnames = firstrow.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    return df
    
    

def save_data(df, database_filename):
    '''
    Function that saves databse in SQL file
    Args: df: df to be saved, database_filename: path to save the database
    Returns: none
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)


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