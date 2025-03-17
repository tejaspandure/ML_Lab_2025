import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    
    titanic_data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries")
    print(titanic_data.head())

    print("Number of passengers are "+str(len(titanic_data)))

    titanic_data.drop("zero", axis=1, inplace=True)
    
    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Value of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head(5))

    print("Values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head(5))

    print("Values of dataset after concatenating new columns ")
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis=1)

    print("Values of dataset after removing irrelevant columns")
    titanic_data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis=1, inplace=True)
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    x.columns = x.columns.astype(str)  # Ensure column names are strings

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

    logmodel = LogisticRegression()

    logmodel.fit(xtrain, ytrain)

    prediction = logmodel.predict(xtest)

    print("Classification report of logistic regression is: ")
    print(classification_report(ytest, prediction))

    print("Confusion matrix of Logistic regression is: ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of Logistic regression is: ")
    print(accuracy_score(ytest, prediction))

def main():
    print("Supervised Machine Learning")
    print("Logistic Regression on Titanic dataset")

    TitanicLogistic()

if __name__ == "__main__":
    main()
