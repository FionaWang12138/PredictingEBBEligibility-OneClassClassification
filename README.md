
Team Name: Machine Intelligence Unit (Team 10)

Team Members:
- Danver Zhao (z5317086)
- Fiona Wang (z5309988)
- Syed Hamza Warisi (z5259749) 
- Mohammed Musa (z5284114)
- Arun Kumar Marndi (z5225111)

Course: COMP9417

Project: TracHack Challenge 22.2

Outline:

The TracHack 22.2 challenge requires participants to use machine learning to make predictions on customers to determine if they are eligible for the Emergency Broadband Benefits Program. This challenge is a one-class classification problem, as the data being held for this problem is a combination of positively labeled and unlabeled data.

Exploratory data analysis and visualisation was conducted. Data processing and cleansing was then performed, followed by an exploration of four different one-class classification approaches. The best method was deemed to be Modified Logistic Regression, and this assessment was determined based on F1 scores of our predictions, where additional feature selection and hyperparameter tuning was executed to finetune the model.

Contents:

1. exploratory_data_analysis.py
- This file includes all code that was used to conduct exploratory data analysis. The code used obtains various plots and diagrams that provide insight into the distribution of the data.

2. data_preprocessing.py
- This file includes all code that was used to clean and compile the multiple datasets into an aggregated dataset, which was then used for analysis in some of the later files and models.

3. isolation_forest.py
- This file includes all code relating to the implementation of the Isolation Forest algorithm. This algorithm makes use of tree structures to detect outliers in our data, and categorises outliers and inliers into the two classes.

4. one_class_svm.py
- This file includes all code relating to the implementation of the One Class Support Vector Machine algorithm. This algorithm is used to learn the decision function for data, and makes use of this for detecting the class of a data point.

5. LinearDiscriminantAnalysis.py
- This files contains the all the relevant codes in implementing the linear discriminant analysis algorithm. Discriminant analysis works by creating one or more linear combinations of predictors, creating a new latent variable for each function.

6. modified_logistic_regression.py
- This file includes all code relating to the implementation of the Modified Logistic Regression algorithm. This algorithm takes a probabilistic approach to estimate the percentage of positive data that is labelled, and uses this to determine which points belong to each class.

7. feature_selection.py
- This file inclues all code to perform feature selection using sklearn.linear_model.LogisticRegression with L1 regularization. The result of this was not used, as it did not improve performance.

8. plotting_feature_weights.py
- This file was used to visualise results from feature_selection.py

All of provided files should be executed in the same order as outlined in the Contents above, to allow for the same results and predictions to be obtained.