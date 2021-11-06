# Traffic Accident Prediction with Machine Learning

In this project, we work with an accidents dataset carrying features related to location, weather conditions, and road infrastructure to predict the severity of accidents. The project is also a part of a
final project task for DSE511 course under the data science and 
the engineering program at the Bredesen Center (University of Tennessee, Knoxville).

## Project Introduction

This project aims to model vehicle accident severity based on weather and road conditions. Once a final model is selected, we plan on performing exploratory factor analysis (or similar methodology) to identify which variables contribute the most to the severity of accidents. Additionally, we explore whether or not accident severity varies significantly by major city within the United States. This is a topic of great significance as vehicular accidents make up approximately 38,000 deaths in the United States each year and cause about 4.4 million hospitalizations.

## Github Workflow

We plan to manage the development of code by having one main branch and three development branches. Each respective member of the team is assigned a development branch. The task is divided using issues and each issues is grouped under milestones. Finally we track all the issues under the project section. Find the issues and the project at the following location:

**[Project Board](https://github.com/DSE511-Project3-Team/DSE511-Project-3-Code-Repo/projects)**, 
**[Current Issues](https://github.com/DSE511-Project3-Team/DSE511-Project-3-Code-Repo/issues)**

## Data

We have selected the dataset titled: “US Accidents (updated) A Countrywide Traffic Accident Dataset (2016 - 2020).” The dataset can be found here: https://www.kaggle.com/sobhanmoosavi/us-accidents

The dataset consists of 1.5 million observations, and each observation has 47 features. And each sample represents an accident that occurred in the United States between 2016 and 2020.

## Methods

We employ ANOVA to find if there is a statistically significant difference between accident severity by city. For the classification task, we use the following methods:

1. Logistic Regression (Baseline)
2. Multinomial Naive Bayes
3. Support Vector Machine

Ensemble methods that we use are:

1. Random Forest
2. XGBoost
3. Adaboost

## Contributing Team Members
|Name     |  Slack Handle   | 
|---------|-----------------|
|Sanjeev Singh|@isanjeevsingh|
|Russ Limber   |@Russtyhub    |
|EonYeon Jo    |@EYJo1        |
