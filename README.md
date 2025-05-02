# Stellar Classification Using Machine Learning: Stars, Quasars, or Galaxies.

This repository holds an attempt to apply machine learning algorithms to classify an Stars, Quasars, and Galaxies based on their spectral characteristics using data from the Sloan Digital Sky Survey [LINK](https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17).


## Overview

The task with this dataset is to classify a Star, Quasar, or Galaxy based on 18 features collected by SDSS. These features include photometric system data, angle information, and various identifiers important to mapping out the night sky. The tabular dataset includes the target feature for a straight-forward supervised-classification approach. I began with two different machine learning algorithms and developed each until reaching an acceptable accuracy metric on each and choosing the best fit model. The two models used were Random Forest and XGBoost. Through hyperparameter tuning and cross validation of both models, I found the best results in the XGBoost model. The best model was able to predict the celestial body within 97% accuracy.


# Summary of Work Done
## Data

-  Type: CSV File


    o	Input: CSV File containing 1 categorical and 16 numerical features that describe spectral characteristics of celestial bodies.

    o	Output: The type of celestial body, Star, Quasar, or Galaxy.

   
-	Size:

    o	The tabular dataset contained a total of 100,000 objects.
 	
-	Instances:
  
    o	The data was split 80/20 where 80% of the data, 80,000 rows, were used to train the     models, and 20% of the data, 20,000 rows, were used to test the models. There were 5     validation folds used to proof each model’s metrics. 


## Preprocessing/Clean up

- During initial visualization of the data I realized there was abundant information collected to locate the area of the sky being surveyed. This lead to a lot of administrative data. Dates, various IDs, specific camera used to capture phenomenon, all these were part of the dataset that posed the risk of overfitting the model. All identification features were removed.

- There was a phenomenon in the photometric system data where instead of using null or leaving a blank, the data collectors decided to place an outlier. This is perhaps to make the missing data easily viewable. This was only a single datapoint and was promptly removed.

- Finally, the target featured string information of “STAR”, “QSO”, and “GALAXY”. Unfortunately, a model cannot directly read this data. I label encoded the target feature designating a number to each class. Galaxy was 0, Quasar was 1, and Star was 2.

## Data Visualization

- Right away I saw that the dataset had an imbalance in the target variable, there was a significant amount of Galaxies compared to Quasars and Stars.

  

- The data was also highly correlated. There were a few features that had very similar correlations.

- With major outliers fixed I still saw a lot of outliers throughout. Quasars in particular had many more than the other classes, this may be because of just how bright they are compared to anything else in the universe.



## Problem Foundation

- The stellar classification dataset was the 17th data release from Sloan Digital Sky Survey. It was a supervised multinomial classification where the input was 17 features of stellar observations, and the output was 1 feature of celestial body type, star, quasar, or galaxy. The dataset was preprocessed by removing identification features to prevent overfitting, removing outliers and encoding specific categories for better model implementation.

- The most developed model was XGBoost, although Random Forest was also attempted. XGBoost is a popular machine learning algorithm because of its accuracy and efficiency. Random Forest was also chosen due to its strength against overfitting, a problem I was keenly aware of.

- Implementing a grid search for hyperparameter tuning on the XGBoost model showed that the model with the highest accuracy score had the following parameters:  {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 300, 'subsample': 0.8}



## Training

- All machine learning model training was performed using Python and sciki-learn libraries on a standard CPU-based environment (Jupyter Notebook and a local machine). The dataset was not particularly big, when it came to visualization and model training timing was insignificant. However, when it came to hyper parameter tuning there were sacrifices made to prioritize time. In cross-validation the machine also took a noticeable amount of time.

- With baseline models I saw a problem of overfitting. The accuracy was close to 100%. I believe this was due to the various Identification features implemented with the goal of mapping the sky in mind. This dataset was not particularly made specifically with classification in mind, more so with mapping everything seen to later be analyzed in various ways.



## Performance Comparison

In this dataset the most important metric is accuracy. The models shown were developed with the goal of accuracy in mind.



- Random Forest Baseline (overfitting):



- XGBoost Baseline:



- XGBoost Cross Validation & Hyper tuning:



## Conclusions

The Sloan Digital Sky Survey has proved to be a good source for well-structured datasets. Their mission of mapping out the night sky for others to analyze is coming to fruition. XGBoost is a popular model for a reason, it is efficient and highly accurate. The most significant insight I can infer from this dataset is the fact that there may be, at least in the area surveyed in this dataset, a significant amount of galaxies compared to quasars, or even stars.

## Future Work

I would like to see a more abstract dataset to provide a more challenging processing period. Instead of getting a dataset from Kaggle, I would like to try a dataset directly from SDSS website. It does seem challenging but well worth the effort.

The models could also be taken further and trained to classify more than 3 classes. They could be used to classify nearby planets or moons, maybe even far off exoplanets and exomoons.

# How to reproduce results

To reproduce the results seen in this repository, navigate to the Model folder where in you’ll find three files, main.csv, functions.py, and XG_Boost.ipynb. That last file will contain the final results seen in this summary.

This project was conducted using a local machine and Jupyter Notebook within Anaconda. The main python libraries used are, matplotlib, pandas, and scikit-learn.



## Overview of Files

There are three main folders in this repository, Mode, Visualization, and Scratch. Model holds the main XGBoost model used, Visualization holds various graphs to better visualize the dataset, and Scratch holds exploratory, difficult to read files.





## Citations

fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17 
https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17.
