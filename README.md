# Disaster Response Pipeline Project

### Project Description
The “Disaster Response Pipeline” is a project to classify messages generated from disaster events. The classification takes place in 36 predefined categories. Examples of the categories are medical_help, medical_products, search_and_rescue, security. Due to the nature of the messages, a message can also be assigned to several categories. With the classification, the message can be assigned to the appropriate disaster relief agencies.

In this project, an ETL and machine learning pipeline will be built. This can make the classification of the messages much easier. A dataset provided by Figure Eight is used to train the classification model. This record contains real messages created and sent during catastrophic events.

Another component of this project is a web app in which a message can be entered, and the result of the classification can be retrieved as a response.

### Installation
The program code runs under Python 3.x. It requires the following Python libraries: 
- numpy
- pandas
- sqlalchemy
- re
- NLTK
- pickle
- sklearn
- plotly
- flask libraries

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### File Descriptions
 The app folder includs the templates folder and "run.py" for the web application.

The data folder includs "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.

The models folder includs "classifier.pkl" and "train_classifier.py" for the Machine Learning model.

README.md file 

Hint: Please ignore the objects "+(database_filepath)" and "DisasterResponse.db"

### Acknowledgements
Thanks to Figure Eight for providing the data for training purposes as part of this project. Also, special thanks to Udacity for the courseware and training.
