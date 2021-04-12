# APP
Application to deploy a time series forecast.

## frameworks 
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Prophet](https://facebook.github.io/prophet/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/)


## Directory
- **data** store the datasets as CSV file.
- **model** store the model's joblib file.
- **templates** store the HTML files and CSS file used by the Flask.
- **app.py** the Flask file where contains the API configuration, routes, etc.
- **data_processing.py** the file contains the class DataProcessing responsible to process the dataset.
- **model.py** the file contains to classes ModelPredict and ModelTrain responsible to predict and train the Prophet model. 
- **test.py** the test file which contains unit-test for machine learning classes.
- **logs.txt** log file as format txt.
