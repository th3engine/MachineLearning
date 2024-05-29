# Decision Tree Classification

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

np.set_printoptions(5)



class Classification():

    def __init__(self,file:str,test_size:float=0.25):
        # Importing the dataset
        dataset = pd.read_csv(file)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=test_size)
        self._scale()
        

    def _scale(self):
        # Feature Scaling
        sc = StandardScaler()
        self.scX_train = sc.fit_transform(self.X_train)
        self.scX_test = sc.transform(self.X_test)

    def _metrics(self,y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        a_sc = accuracy_score(self.y_test, y_pred)
        return cm, a_sc


    def decision_tree(self,criterion:str='entropy'):
        classifier = DecisionTreeClassifier(criterion=criterion)
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)
    
    def k_nn(self, n_neighbors:int=5,metric:str='minkowski',p:int=2):
        classifier = KNeighborsClassifier(n_neighbors,metric=metric,p=p)
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)
    
    def kernel_svm(self,kernel:str):
        classifier = SVC(kernel = kernel)
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)
    
    def logistic_regression(self):
        classifier = LogisticRegression()
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)

    def naive_bayes(self):
        classifier = GaussianNB()
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)

    def rand_forest(self, n_estimators:int=10, criterion:str = 'entropy'):
        classifier = RandomForestClassifier(n_estimators,criterion=criterion)
        classifier.fit(self.scX_train, self.y_train)
        y_pred = classifier.predict(self.scX_test)
        return self._metrics(y_pred)
    
    def test_all(self):
        
        results = {
            "Decision Tree":self.decision_tree()[1],
            "K Nearest Neighbors":self.k_nn()[1],
            "Kernel SVM (Linear)":self.kernel_svm('linear')[1],
            "Kernel SVM (RBF)":self.kernel_svm('rbf')[1],
            "Logistic Regression":self.logistic_regression()[1],
            "Naive Bayes":self.naive_bayes()[1],
            "Random Forest":self.rand_forest()[1],
        }

        return pd.Series(results,name="Accuracy Scores")
    
test = Classification("Data.csv")
print(test.test_all().idxmax())

