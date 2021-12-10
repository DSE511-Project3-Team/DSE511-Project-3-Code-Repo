# Traffic Accident Prediction with Machine Learning

In this project, we work with an accidents dataset carrying features related to location, weather conditions, and road infrastructure to predict the severity of accidents. The project is also a part of a
final project task for DSE511 course under the data science and 
the engineering program at the Bredesen Center (University of Tennessee, Knoxville).

## Project Introduction

This project aims to model vehicle accident severity based on weather and road conditions. Once a final model is selected, we plan on performing exploratory factor analysis (or similar methodology) to identify which variables contribute the most to the severity of accidents. Additionally, we explore whether or not accident severity varies significantly by major city within the United States. This is a topic of great significance as vehicular accidents make up approximately 38,000 deaths in the United States each year and cause about 4.4 million hospitalizations.

## Data

We have selected the dataset titled: “US Accidents (updated) A Countrywide Traffic Accident Dataset (2016 - 2020).” The dataset can be found here: https://www.kaggle.com/sobhanmoosavi/us-accidents

The dataset consists of 1.5 million observations, and each observation has 47 features. And each sample represents an accident that occurred in the United States between 2016 and 2020.

### Accessing Data:
1. The full raw data carrying 1.5 million observations can be downloaded from [here](https://www.dropbox.com/s/mdw2asjrh8bm038/US_Accidents_Dec20_updated.csv?dl=1).

2. We build our analysis after performing downsampling for six cities: Phoenix, Los Angeles, New York, Philadelphia, Houston, and Chicago. This dataset can be found inside the folder: /data/raw/accident_data.csv. To regenerate this dataset execute the following command.

```
python main.py data
```

Note that the above command also creates an imputed dataset which we use to do further downstream work. Find it inside /data/processed/imputed.pkl. 

## Generate Results

1. To generate the results from our selected models execute the following command. But do make sure you have the neccessary pickle file under /data/processed/imputed.pkl folder. See "Accessing Data" section to know how you can create this file if it does not exists.

```
python main.py results
```

2. To see the results from the hyperparameter tuning execute the following command. Again, make sure you have the neccessary pickle file under /data/processed/imputed.pkl folder. See "Accessing Data" section to know how you can create this file if it does not exists. But be warned that, it'll take hours to complete this execution.

```
python main.py tune
```

## Methods

We employ ANOVA to find if there is a statistically significant difference between accident severity by city. For the classification task, we use the following methods:

1. Logistic Regression (Baseline)
2. Multinomial Naive Bayes

Ensemble methods that we use are:

1. Random Forest
2. XGBoost
3. Adaboost
4. Gradient Boosting

## Github Workflow

We plan to manage the development of code by having one main branch and three development branches. Each respective member of the team is assigned a development branch. The task is divided using issues and each issues is grouped under milestones. Finally we track all the issues under the project section. Find the issues and the project at the following location:

**[Project Board](https://github.com/DSE511-Project3-Team/DSE511-Project-3-Code-Repo/projects)**, 
**[Current Issues](https://github.com/DSE511-Project3-Team/DSE511-Project-3-Code-Repo/issues)**

## Repository Structure

    ├── README.md          <- The top-level README carrying the project description and organization.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original data that we use for further processing.
    │
    ├── docs
    │   ├── images         <- Folder saving generated images for report and presentation.
    │   └── reports        <- Folder carrying reports and presentations submitted during the project.
    │
    ├── notebooks          <- Folder carrying Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code used in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions.
    │   ├── preprocessing  <- Scripts to perform train/test split, feature selection, feature 
    |   |                     extraction etcetra.
    |   |── results        <- Folder carrying script to generate results from our tuned models.    
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
    │
    └── main.py            <- Script that will run all the necessary code to generate the results.
    

## Team Members
|Name     |  Github Handle   | 
|---------|-----------------|
|Sanjeev Singh|@isanjeevsingh|
|Russ Limber   |@Russtyhub    |
|EonYeon Jo    |@EYJo1        |
