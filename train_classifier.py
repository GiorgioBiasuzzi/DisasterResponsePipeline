#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[9]:


# import libraries
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


# In[10]:


# load data from database
engine = create_engine('sqlite:///MyProject.db')
df = pd.read_sql_table('TblHelpRequests', engine)  


X = df['message']
Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)


# ### 2. Write a tokenization function to process your text data

# In[11]:


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


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[12]:


pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[13]:


#Split between train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, random_state=42)
#Train pipeline
pipeline.fit(Xtrain, ytrain)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[14]:


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

report_scores(pipeline, Xtest, ytest) 


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[15]:


parameters = {'clf__estimator__n_estimators': [20, 50]}


# In[16]:


cv = GridSearchCV(pipeline, parameters)


# In[17]:


cv.fit(Xtrain, ytrain)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[18]:


#print('========================TEST SET=========================')
report_scores(cv, Xtest, ytest)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[19]:


pipelineimproved = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

parameters_improved = {'clf__estimator__learning_rate': [0.1, 0.3],
                       'clf__estimator__n_estimators': [100, 200]}

                            
cvimproved = GridSearchCV(pipelineimproved, param_grid=parameters_improved)
cvimproved.fit(Xtrain, ytrain)
                            
report_scores(cvimproved, Xtest, ytest)


# ### 9. Export your model as a pickle file

# In[20]:


pickle.dump(cvimproved, open('model.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




