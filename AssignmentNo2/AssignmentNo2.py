import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Predictor():
    # Load data,
    data = pd.read_csv("HeadBrain.csv")

    print("Size of data set",data.shape)

    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values

    # Least Squares method
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    print("Mean of Independent variable X:", mean_x)
    print("Mean of dependent variable Y:", mean_y)
    n = len(X)

    numerator = 0
    denominator = 0

    # Equation of the line is y = mx + c

    for i in range(n):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denominator += (X[i] - mean_x) ** 2

    m = numerator / denominator

    # c = y' - mx'
    c = mean_y - (m * mean_x)

    print("Slope of regression line is:", m)  # Example slope, e.g. 0.4
    print("Y-intercept of regression line is:", c)  # Example y-intercept, e.g. 2.4

    max_x = np.max(X)+100
    min_x = np.min(X)-100

    # Display plotting of above points
    x = np.linspace(min_x,max_x,n)  # Broader range for x values (from 0 to 6)
    y = c + m * x  # Calculating y values based on the regression line

    plt.plot(x, y, color='green', label='Regression Line')
    plt.scatter(X, Y, color='orange', label='Scatter Plot')

    plt.xlabel("X - Head size in cm3")
    plt.ylabel("Y - Brain weight in gram")

    plt.legend()
    plt.show()

def main():
    print("Supervised Machine Learning")
    print("Linear Regression")
    Predictor()

if __name__ == "__main__":
    main()
