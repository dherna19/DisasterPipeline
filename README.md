# DisasterPipeline
Udacity Data Scientist Nanodegree Project

# Project Motivation
For this Udacity project, I will analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

#ETL Pipeline
The Python script, process_data.py, is data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#ML Pipeline
The Python script, train_classifier.py, is a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#Flask Web App

 run.py: A web app that does the following:
- includes data visualizations using Plotly
- The web app successfully uses the trained model to input text and return classification results.
- When a user inputs a message into the app, the app returns classification results for all 36 categories
- run.py

# Installation
This project requires Python 3.x and the following Python libraries installed:
- Pandas
- Sci-kit learn
- NumPy
- Seaborn
- Matplotlib

# Data
Data was provided by Figure8 and Udacity 
File Descriptions
Jupyter Notebook, CSV files:
-	disaster_categories.csv: has the 36 categories of the type of message it can be: shelter, food, etc.
-	disaster_messages.csv: includes a list of all the messages sent to disaster organizations or via social media by people affected by a natural disaster


# Acknowledgement
This dataset is part of Airbnb Inside, and the original source can be found here.
I have answered a lot of my questions relating to data cleaning and preparation using different sources from StackOverFlow and other websites:

