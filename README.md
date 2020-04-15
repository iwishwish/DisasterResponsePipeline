# Disaster Response Pipeline


### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)


## Installation <a name="installation"></a>
1. Run following command to update scikit-learn to version 0.22.2.post1.     
`pip3 install -U scikit-learn`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/ 



## Project Motivation<a name="motivation"></a>

    

## File Descriptions <a name="files"></a>
- **app**    
  &ensp;| - **template**    
  &ensp;|   &ensp;  |- **master.html**  &nbsp; *main page of web app*    
  &ensp;|    &ensp; |- **go.html**  &nbsp; *classification result page of web app*     
  &ensp;|- **run.py**   &nbsp;*Flask file that runs app*  

- **data**    
  &ensp;|- **disaster_categories.csv**    &nbsp;*data to process (Not submitted to repository. Should put file in this path when running ETL.)*    
  &ensp;|- **disaster_messages.csv**   &nbsp;*data to process (Not submitted to repository. Should put file in this path when running ETL.）*   
  &ensp;|- **process_data.py**  &nbsp;*ETL pipeline*  
  &ensp;|- **DisasterResponse.db**   &nbsp;*database to save clean data to*  

- **models**    
  &ensp;|- **train_classifier.py**  &nbsp;*ML pipeine*  
  &ensp;|- **classifier.pkl**   &nbsp;*saved model*     

- **README.md**    
    




