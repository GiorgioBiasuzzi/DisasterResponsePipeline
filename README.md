# Disaster Response Pipeline
Disaster Response Pipeline
Project Components


The objective of this project is to find a data set containing real messages that were sent during disaster events. 
A machine learning pipeline will be built to categorize these events so that the messages can be sent to an appropriate disaster relief agency.


There are two components in this project.

1. ETL Pipeline
In a Python script, process_data.py, is stored a data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, is stored a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file





Data Source: Figure Eight through Udacity Nanodegree course
