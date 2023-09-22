# Kaggle-Loan-Approval

# Introduction
This project uses the following loan approval dataset from Kaggle https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
In this project we develop a machine learning classifcation model using Scikit-learn in Python to predict the approval status of a loan with 98% accuracy and f1-score=0.98. 

# Exploratory Data Analysis
We initially check the data for missing values, which this dataset has none, get a list of the features and get the summary statistics for the data.

Dependent Variable : Loan Status

Features:
* Number of Dependents
* Education (Graduate or Non-Graduate)
* Employment Status
* Annual Income
* Loan Amount
* Loan Term
* Credit Score
* Residential Asset Value
* Commercial Asset Value
* Luxury Asset Value
* Bank Asset Value

From this we then procedd to carry out some Exploratory Data Analysis using seaborn to visualise the data and understand the relations between the variables.
We generate bar graphs, histograms and a correlation heatmap.
Some of which can be seen below:

![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/c5298631-7623-4c05-bad4-eda62ab0756a)  ![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/97ccb04a-6888-4d99-a88c-5b6d380a0b1e)

![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/3f4e60b7-0271-4c2a-a0b3-f88ad8d75a44)   ![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/b06af6da-7117-4c7e-85c0-e432cdb40c92)

![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/fcdad1e8-6a70-44e1-98cb-d505ccfc859c)

# Preprocessing Data
Applying label encoding to the categorical variables which are Education, Employment and the dependent variable loan status we can convert these features into dummy variables.
Then we train-test split the data and apply StandardScaler to the numerical variables to normalise them.

# Model Builiding and Tuning
We initially construct and evaluate a variety of classificaion algorithms with default parameters :
* Random Forest
* Support Vector Machine
* Logistic Regression
* Multilayered Perceptron

Then applying GridSearchCV we fine-tune the hyperparameters of the Random Forest, Support Vector Machine, and MLP so as to optimise the f1-score.

Best f1-scores:
* Random Forest - 0.98
* SVM - 0.93
* MLP - 0.97

# Final Model
In the end the Random Forest classifier is chosen and its classification report can be seen below:

![image](https://github.com/PrishalRadia/Kaggle-Loan-Approval/assets/140926795/34ff11c9-26bd-4e5f-9c2c-0d5320af9681)


# References 

* https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
* https://scikit-learn.org/stable/index.html
