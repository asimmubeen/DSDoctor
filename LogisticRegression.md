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


