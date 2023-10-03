import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('dulieu.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

X_train = dt_Train.iloc[:, :7]
y_train = dt_Train.iloc[:, 7]
X_test = dt_Test.iloc[:, :7]
y_test = dt_Test.iloc[:, 7]
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
y = np.array(y_test)

print("Coefficent of determination: %.9f" % r2_score(y_test, y_pred))

print("Thuc te Du doan Chenh lech")
for i in range(0, len(y)):
    print("%.2f" % y[i], "  ", y_pred[i], "  ", abs(y[i]-y_pred[i]))

