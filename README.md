# Credit_Risk_Analysis
## Overview
Building and evaluating several machine learning models in the branch of **Supervised Learning** to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.
## Purpose
The purpose of this analysis is to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on `Supervised Learning` using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk.
To complete this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. These algorithms include `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier`.
###  Resources
* Data Source: LoanStats_2019Q1.csv
* Software: Python 3.7.9, Jupyter Notebook 6.0.3
* Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
## Results
The results for the six machine learning models including their respective balanced accuracy, precision, and recall scores are as follows:
### Naive Random Oversampling
<img width="459" alt="Screen Shot 2022-10-07 at 9 52 43 PM" src="https://user-images.githubusercontent.com/107584361/194689866-84a3f37c-faa7-4758-a49c-309e0d87239e.png">
<img width="722" alt="Screen Shot 2022-10-07 at 9 52 49 PM" src="https://user-images.githubusercontent.com/107584361/194689869-ac22669b-1395-4a07-bb4f-6436a94bf2f8.png">

* Balanced accuracy score: 65%.
* The "High Risk" precision rate was only 1% with the recall at 63% giving this model an F1 score of 2%.
* "Low Risk" had a precision rate of 100% and recall at 67%.

### SMOTE Oversampling
<img width="732" alt="Screen Shot 2022-10-07 at 10 22 31 PM" src="https://user-images.githubusercontent.com/107584361/194690141-a03cd54a-cc61-4f19-9802-c2ef02c2d800.png">

* Balanced accuracy score: 65%.
* The "High Risk" precision rate was only 1% with the recall at 64% giving this model an F1 score of 2%.
* "Low Risk" had a precision rate of 100% and recall at 66%.

### Cluster Centroids (Undersampling)
<img width="749" alt="Screen Shot 2022-10-17 at 8 10 50 PM" src="https://user-images.githubusercontent.com/107584361/196326864-fd8e312a-3f11-4bfc-8e5a-1a9091dc088a.png">

* Balanced accuracy score: 53%.
* The "High Risk" precision rate was only 1% with the recall at 61% giving this model an F1 score of 1%.
* "Low Risk" had a precision rate of 100% and recall at 45%.

### SMOTEENN (Combination Sampling)
<img width="811" alt="Screen Shot 2022-10-07 at 10 31 47 PM" src="https://user-images.githubusercontent.com/107584361/194690414-a9554c7d-3909-48c9-b135-af4e35f1f790.png">

* Balanced accuracy score: 63.75%.
* The "High Risk" precision rate was only 1% with the recall at 70% giving this model an F1 score of 2%.
* "Low Risk" had a precision rate of 100% and recall at 57%.

### Balanced Random Forest Classifier
<img width="1033" alt="Screen Shot 2022-10-07 at 10 36 18 PM" src="https://user-images.githubusercontent.com/107584361/194690784-0dbf9c93-2540-4c00-9f8a-12299e380e9c.png">

* Balanced accuracy score: 78.77%.
* The "High Risk" precision rate was only 4% with the recall at 67% giving this model an F1 score of 2%.
* "Low Risk" had a precision rate of 100% and recall at 91%.

### Easy Ensemble AdaBoost Classifier
<img width="1017" alt="Screen Shot 2022-10-07 at 10 40 13 PM" src="https://user-images.githubusercontent.com/107584361/194690948-f713accc-1838-41a3-8283-145217fba940.png">

* Balanced accuracy score: 92.54%.
* The "High Risk" precision rate was only 7% with the recall at 91% giving this model an F1 score of 14%.
* "Low Risk" had a precision rate of 100% and recall at 94%.

## Summary
In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 92.54% and a 7% precision rate when predicting "High Risk candidates. The sensitivity rate (recall) was also the highest at 91% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

The only drawback of this model is that with low precision, a lot of low risk credits are still falsely detected as high risk. In this scenario though, it is better for the model to have greater senstivity than precision. So no model is recommended.
