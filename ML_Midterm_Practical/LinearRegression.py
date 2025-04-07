import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("HeadBrain.csv")
X = df.drop('Head Size(cm^3)',axis=1)
y = df['Brain Weight(grams)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Square Error: ",mean_squared_error(y_test,y_pred))
print("Mean Abosolute Error: ",mean_absolute_error(y_test,y_pred))
print("R2_score: ",r2_score(y_test,y_pred))

print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("Y_train: ",y_train.shape)
print("Y_test: ",y_test.shape)

