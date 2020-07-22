import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from typing import Tuple, List
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pickle


def load_data(database_filepath):
    """Load & merge messages & categories datasets
    
    inputs:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    outputs:
    df: dataframe. Dataframe containing merged content of messages & categories datasets.
    """

    #Load Messages Dataset
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # split into features and target
    X = df["message"]
  
    # Making DataFrame with relevant categories
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    Y['related']= Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    print(Y.shape)

    return X, Y, category_names


def tokenize(text):
    '''
    Function to tokenize strings
    Args: Text string
    Returns: List of tokens
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    return [lemmatizer.lemmatize(x).lower().strip() for x in tokens]


def build_model():
    '''
    Function for pipeline and GridSearch
    Args: None
    Returns: Model
    '''
 #build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
#parameters for gridsearch
    parameters = {'clf__estimator__n_estimators': [50, 100],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='accuracy',verbose= 10,n_jobs =-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    score = model.score(X_test, Y_test)
    print('Accuracy: {:0.4f}'.format(score))
    Y_pred = model.predict(X_test)
    print(Y_pred.shape)
    
  
      

def save_model(model, model_filepath):
    '''
    Function for saving the model as picklefile
    Args: Model, filepath
    Returns: Nothing. Saves model to pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()