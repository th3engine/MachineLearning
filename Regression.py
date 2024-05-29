# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Regression:

    
    def __init__(self, file: str, test_size: float = 0.2):
    # Importing the dataset
        try:
            dataset = pd.read_csv(file)
        except FileNotFoundError:
            print("File not found.")
            return
        
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 0) # Splitting the dataset into the Training set and Test set
        np.set_printoptions(precision=2) # prints numpy all arrays to 2 decimal places



    # Decision Tree Regression
    def decison_tree(self):
    
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train) # Training the Decision Tree Regression model on the Training set
        y_pred = regressor.predict(self.X_test) # Predicting the Test set results

        table = np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1)
        return r2_score(self.y_test, y_pred), table
    
    # Multiple Linear Regression
    def multiple_linear(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.X_test)
        table = np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1)
        return r2_score(self.y_test, y_pred), table

    # Polynomial Regression
    def polynomial(self, degree:int = 4):
        poly_reg = PolynomialFeatures(degree)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)

         # Predicting the Test set results
        y_pred = regressor.predict(poly_reg.transform(self.X_test))
        table = np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1)

        return r2_score(self.y_test, y_pred), table


    # Random Forest Regression
    def rand_forest(self, n_estimators: int = 10):
        regressor = RandomForestRegressor(n_estimators, random_state = 0)
        regressor.fit(self.X_train, self.y_train)
   
        y_pred = regressor.predict(self.X_test)
        table = np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1)
        return r2_score(self.y_test, y_pred),table


    # Support Vector Regression (SVR)
    def support_vector(self):
        y_train = self.y_train.reshape(len(self.y_train),1) # turn y_train into a column so it can be feature scaled
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(self.X_train)
        y_train = sc_y.fit_transform(y_train)

        # Training the SVR model on the Training set

        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train.ravel())
        # Predicting the Test set results
        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test)).reshape(-1,1))
        
        table = np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1)
        return r2_score(self.y_test, y_pred),table

    def test_all(self):

        scores = {
            "Decision Tree":self.decison_tree()[0],
            "Multiple Linear":self.multiple_linear()[0],
            "Polynomial":self.polynomial()[0],
            "Random Forest": self.rand_forest()[0],
            "Support Vector": self.support_vector()[0],
        }
        return pd.Series(scores,name="R2 Scores")



test = Regression("Data.csv")
print(test.test_all().idxmax())







