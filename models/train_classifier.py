import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import type_of_target
from sklearn.svm import SVC

import pickle


def load_data(database_filepath):
    """
    INPUT:
    database_filepath  
    
    OUTPUT:
    X - messages 
    y - categories of the messages
    my_columns - category names for y
    """

    engine = create_engine('sqlite:///' +database_filepath)
    df = pd.read_sql('DisasterMessages', engine)
    
    X = df.message
    y = df.iloc[:, 4:]
    my_columns = y.columns
    
    return X, y, my_columns


def tokenize(text):
    """
    INPUT: 
    raw text, afterwards normalized, stop words removed, lemmatized
    
    OUTPUT: 
    tokenized text
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Handling URLs
    url_find = re.findall(url_regex, text)
    for u in url_find:
        text = text.replace(u, 'urlplaceholder')
    
    # Normalize text
    words = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
    # Tokenizing
    words = word_tokenize(words)
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    words = [w for w in words if w not in stop_words]
            
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
   
    return words


def build_model():
    """ 
    Machine learnig pipeline
    
    INPUT: 
    classifier, if no input then default = AdaBoostClassifier
    
    OUTPUT:
    model
    """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize))
    , ('tfidf', TfidfTransformer())
    , ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    parameters = {
    'clf__estimator__n_estimators': [20, 40],
    'clf__estimator__learning_rate': [0.2, 0.4]
    }

    model = GridSearchCV(pipeline, param_grid = parameters)
        
    return model


def evaluate_model(model, X_test, y_test, my_columns):
    """
    INPUT:
    model from AdaBoostClassifier and GridSearchCV
    X_test,  messages.
    y_test,  categories of the messages
    my_columns,  category_names for y 
    
    OUTPUT:
    no return
    print scores: precision, recall, f1-score for each category
    """

    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred, target_names = my_columns))
    
        
def save_model(model, model_filepath):
    """
    INPUT:
    model 
    filepath
    
    OUTPUT:
    none
    """
    
    with open(model_filepath, 'wb') as pick_file:
        pickle.dump(model, pick_file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, my_columns = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, my_columns)

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