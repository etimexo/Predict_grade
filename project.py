import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
df = pd.read_csv("student_mat.csv",  sep=";")
# df = df[["sex", "G1", "G2", "G3", "studytime", "failures", "absences", "internet", "guardian", "freetime", "romantic"]]
df = df[["G1", "G2", "G3", "studytime", "failures"]]

predict = "G3"
X = np.array(df.drop(columns=[predict]))
y = np.array(df[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
predictions = linear.predict(x_test) # Gets a list of all predictions
print(acc)

for x in range(len(predictions)):
    # print(predictions[x], x_test[x], y_test[x])
    pass

# print(acc)
# print('Coefficient: \n', linear.coef_) # These are each slope value
# print('Intercept: \n', linear.intercept_) # This is the intercept

# print(df.head())