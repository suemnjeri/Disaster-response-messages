import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, on='id')
    
    return df
    


def clean_data(df):
    categories_df = df['categories'].str.split(';', expand=True)
    row = categories_df.loc[0]
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories_df.columns = category_colnames
    for column in categories_df:
        categories_df[column] = categories_df[column].str.split('-', expand=True)[1]
        categories_df[column] = pd.to_numeric(categories_df[column])
    df = df.drop('categories', axis=True)
    df = pd.concat([df, categories_df], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_details', engine, index=False)


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