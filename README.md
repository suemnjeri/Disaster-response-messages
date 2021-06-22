# Disaster Response Pipeline Project

### Introduction
This project is about deciphering messages sent during disasters and allocate them into appropriate categories.
The data was provided by Figure Eight and it classifies disaster messages from various sources such as Twitter and 
texk messages into 36 categories.

### Installation
The following Libraries are required for the pproject.
* Python 3
* Anaconda
* Pandas, Numpy, Re, Sqlalchemy, Pickle, Sklearn, and NLTK

### Project steps
```
1. ETL Pipeline (Extract, Transform and Load)
- Read the dataset. There are 2 CSV files: 'messages.csv' and 'categories.csv'
- Merge the two datasets
- Clean the data using Pandas
- Save the data into an SQLite database.
2. ML Pieline (Machine Learning)
- Split the data into training and test set.
- Create a model using NLTK, scikit-learn's Pipeline and GridSearchCV
- Use the final model to classify a 'message' into 36 categories.
3. Flask App
- Display the results into a Flask web app
```

### Files
Below is the structure of the files in the repository
 ```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 
```


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


