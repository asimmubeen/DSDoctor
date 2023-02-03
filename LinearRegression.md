# Linear Regression
Linear regression is a statistical method used to model the linear relationship between a dependent variable and one or more independent variables. The basic idea is to use the independent variables to explain or predict the dependent variable. The relationship between the variables is modeled using a linear equation, where the parameters of the equation are estimated from the data. The general form of the linear regression equation is:

$$y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$$

where y is the dependent variable, $x_1, x_2, ..., x_n$ are the independent variables, and $b_0, b_1, b_2, ..., bn$ are the coefficients of the model. These coefficients represent the effect of each independent variable on the dependent variable, and they can be positive, negative, or zero. The goal of linear regression is to find the values of the coefficients that minimize the difference between the predicted values of the dependent variable and the actual values in the data.

Linear regression is a simple and powerful method for modeling linear relationships between variables, and it is widely used in many fields, including economics, finance, and engineering. However, it is limited to modeling linear relationships and may not be appropriate for datasets where the relationship between the variables is more complex. In these cases, more sophisticated methods, such as non-linear regression or machine learning algorithms, may be required.

<p align="center"> <img src="https://user-images.githubusercontent.com/24811295/216652104-a52f07ec-e361-4b10-b960-f96c65e08a7f.png" height="300" width="300" > </p>

## Python Code
Here is an example of linear regression in Python using the scikit-learn library:

In the following example, the data is loaded from a CSV file using pandas, and then split into a training set and a test set using the train_test_split function. The linear regression model is created using the LinearRegression class, and then fitted to the training data using the fit method. Finally, the model is used to make predictions on the test data using the predict method, and the mean squared error of the predictions is computed and printed.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("data.csv")

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2)

# Create a linear regression object
lin_reg = LinearRegression()

# Fit the linear regression model to the training data
lin_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lin_reg.predict(X_test)

# Evaluate the performance of the model
mse = np.mean((y_pred - y_test)**2)
print("Mean Squared Error:", mse)


```
