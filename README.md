# UC Berkeley Intro to Machine Learning Final Project: ANES Election Data Classification
 
## Description

This repository contains the code and data for a 2-week final project in UC Berkeley's Intro to Machine Learning course, where I worked with the [American National Election Survey data](https://electionstudies.org/data-center/2016-time-series-study/). This survey is done pre and post every American election and asks questions about various social and political topics. For the project, we used random forests and AdaBoost ensemble learners to classify whether a person voted Democratic or Republican for the Presidential election in 2016 using their answers to various apolitical questions from the survey. 

For the complete methodology and analysis, refer to [the final report](CS289-Final-Report.pdf) submitted by Kevin Sun and Axel Amzallag. 

## Code
+ **dataprocessing.py** contains functions used to import and clean the full data, including subsetting of survey responses by category and initial cleaning and transformations for model training and fitting
+ **fitting.py** contains functions for fitting random forests and AdaBoost classifiers to each subset of survey responses, along with hyperparameter selection using k-fold cross validation
+ **experimental.py** is a script used to generate the outputs using the functions of *dataprocessing* and *fitting*. 
