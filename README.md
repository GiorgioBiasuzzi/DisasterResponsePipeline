# Disaster Response Pipeline
Disaster Response Pipeline
Project Components


The objective of this project is to find a data set containing real messages that were sent during disaster events. 
A machine learning pipeline will be built to categorize these events so that the messages can be sent to an appropriate disaster relief agency.


There are two components in this project.

1. ETL Pipeline
In a Python script, process_data.py (or Jupyter Notebook "ETL Pipeline Preparation"), is stored a data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py (or Jupyter Notebook "ML Pipeline Preparation"), is stored a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file





Data Source: Figure Eight through Udacity Nanodegree course


# Flask README

# Disaster Response Pipeline Project


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
