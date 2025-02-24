import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Predictor():
    # Load data
    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    print("Values of independent variables X:", X)
    print("Values of dependent variable Y:", Y)

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

    # Display plotting of above points
    x = np.linspace(0, 6, 100)  # Broader range for x values (from 0 to 6)
    y = c + m * x  # Calculating y values based on the regression line

    plt.plot(x, y, color='green', label='Regression Line')
    plt.scatter(X, Y, color='red', label='Scatter Plot')

    plt.xlabel("X - Independent variable")
    plt.ylabel("Y - Dependent variable")

    plt.legend()
    plt.show()

def main():
    print("Supervised Machine Learning")
    print("Linear Regression")
    Predictor()

if __name__ == "__main__":
    main()
