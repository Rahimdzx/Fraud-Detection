# Fraud Detection Using Machine Learning
![image](https://github.com/user-attachments/assets/da8b1817-2ae6-4998-9020-6027b4fb4359)

This project is a fraud detection system that uses a machine learning model trained to classify transactions as fraudulent or non-fraudulent. The dataset used contains transactions made by credit cardholders, with features transformed using PCA (Principal Component Analysis).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Technologies Used](#technologies-used)


## Overview
The goal of this project is to predict whether a credit card transaction is fraudulent or not. The dataset is highly imbalanced, with only a small fraction of transactions labeled as fraudulent.

We use a machine learning model based on Support Vector Classification (SVC), along with hyperparameter tuning using GridSearchCV, to improve the model's performance. The features in the dataset have been transformed using Principal Component Analysis (PCA).

## Dataset
The dataset contains the following features:
- **Time**: The time in seconds between the current transaction and the first transaction in the dataset.
- **Amount**: The transaction amount.
- **V1-V28**: The principal components obtained through PCA.
- **Class**: The label (1 for fraudulent transactions, 0 for non-fraudulent).

The dataset is highly imbalanced, with only 0.172% of the transactions labeled as fraudulent.

## Model
The model used is **Support Vector Classification (SVC)**. To optimize the performance of the model, **GridSearchCV** was used to perform hyperparameter tuning. 

The pipeline includes:
- **StandardScaler**: To normalize the feature values.
- **SMOTE**: To handle class imbalance by generating synthetic samples.
- **SVC**: To classify the transactions as fraudulent or non-fraudulent.

GridSearchCV was used to find the best combination of hyperparameters for the SVC model.

## Technologies Used
- **Python**: Programming language
- **Flask**: Web framework for creating the API
- **scikit-learn**: For machine learning algorithms and tools
- **pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
