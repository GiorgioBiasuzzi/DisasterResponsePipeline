#import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import nltk
import pickle
nltk.download(['punkt', 'wordnet', 'stopwords'])
import sys


def load_data(database_filepath):
    """
    Function that loads messages and categories from database using database_filepath as a filepath and sqlalchemy as library
    Returns two dataframes X and Y
    """
    engine = create_engine('sqlite:///' + str (database_filepath))
    df = pd.read_sql ('SELECT * FROM TblHelpRequests', engine)
    X = df ['message']
    y = df.iloc[:,4:]

    return X, y


def tokenize(text):
    """
    This function splits text into words and returns the base of the word
    It will be used when building the ML pipeline later in the code
    The argument is text which is feeding from the message column of the df dataframe
    """
    #Transform everything in lower case
    text = text.lower()
    #Tokenize words
    tokens = nltk.word_tokenize(text)
    #Remove stopwords to avoid potential inconsequential tokens
    stop_words = stopwords.words("english")
    word = [word for word in tokens if word not in stop_words]
    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
              
    return lemmatized

def build_model():
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    """
    # model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # hyper-parameter grid
    parameters = {'clf__estimator__n_estimators': [20, 50]}

    # create model
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def report_scores(model, Xtest, ytest):
    """This function applies the sklearn classification_report
       on the required model (in our case pipeline) and returns 
       the f1-score and the precision of the model using the model
       and the test series as arguments
    """
    
    ypreds = model.predict(Xtest)
    
    for i, col in enumerate(ytest):
        print('------------------------------------------------------\n')
        print('FEATURE NAME: {}\n'.format(col))
        print(classification_report(ytest[col], ypreds[:, i]))
    pass


def save_model(model, model_filepath):
    """ Saving model using pickle  """
    
    pickle.dump(cvimproved, open('model.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(Xtrain, ytrain)
        
        print('Evaluating model...')
        report_scores(model, Xtest, ytest)

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