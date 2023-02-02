# Logistic Regression
Logistic regression is the workhorse in supervided machine learning classification problem. It is popular for its role as binary classification, however it can be used as multiclass classification. Regression??? No no, it is classifiation technique. Let us learn some background theory if we may. I think we should! you will thank me later.

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.

The logistic regression model is a mathematical equation that defines the relationship between the independent variables and the binary outcome. The model uses a logistic function to calculate the probability of the default class (e.g. 0) or the positive class (e.g. 1). The logistic regression algorithm is used to train the model on a given dataset by adjusting the coefficients of the independent variables to minimize the difference between the predicted outcome and the actual outcome.
## The Sigmoid / Logistic Function
The sigmoid function, also known as the logistic function, is a mathematical function that maps any input value to a value between 0 and 1. It is defined as:

f(x) = 1 / (1 + e^(-x))

where e is the base of the natural logarithm. In logistic regression, the sigmoid function is used to model the probability of a binary outcome as a function of the input variables. The output of the sigmoid function is interpreted as the probability of the positive class, given the values of the independent variables.

![image](https://user-images.githubusercontent.com/24811295/216444943-2e5271cb-76a9-4ee7-8a80-1fe5ef5b422e.png)

In logistic regression, the sigmoid function is used to model the relationship between the independent variables and the binary dependent variable. The basic idea is to use the independent variables to calculate a linear combination of their values, which is then transformed by the sigmoid function to produce a probability value between 0 and 1. This transformed value can then be used to make predictions about the binary dependent variable.

More specifically, the logistic regression model is based on the following equation:

p(y=1|x) = 1 / (1 + e^-(b0 + b1 * x1 + b2 * x2 + ... + bn * xn))

where p(y=1|x) is the predicted probability of the positive class (y = 1), given the values of the independent variables x1, x2, ..., xn. b0, b1, b2, ..., bn are the coefficients of the model, which are estimated from the training data using maximum likelihood estimation.

Once the coefficients have been estimated, the logistic regression model can be used to make predictions for new data by plugging in the values of the independent variables and calculating the corresponding probability of the positive class. If the calculated probability is greater than or equal to 0.5, the prediction will be the positive class (y = 1), otherwise it will be the negative class (y = 0). The sigmoid function is used to ensure that the predicted probability values always fall within the range of 0 and 1, which is appropriate for modeling binary outcomes.

## Python Code

Here is a simple example of logistic regression in Python using the scikit-learn library:

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

##### Load the data
df = pd.read_csv("data.csv")

##### Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2)

##### Create a logistic regression object
logistic_reg = LogisticRegression()

##### Fit the logistic regression model to the training data
logistic_reg.fit(X_train, y_train)

##### Make predictions on the test data
y_pred = logistic_reg.predict(X_test)

##### Evaluate the performance of the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


In this example, the data is loaded from a CSV file using pandas, and then split into a training set and a test set using the train_test_split function. The logistic regression model is created using the LogisticRegression class, and then fitted to the training data using the fit method. Finally, the model is used to make predictions on the test data using the predict method, and the accuracy of the predictions is computed and printed.



