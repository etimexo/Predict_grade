import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# style.use("ggplot")

df = pd.read_csv("student_mat.csv", sep=";")

predict = "G3"

df = df[["G1", "G2", "absences","failures", "studytime","G3"]]
df = shuffle(df) # Optional

x = np.array(df.drop(columns=[predict]))
y =np.array(df[predict])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Train the model 30 times for the best score and pass as the one we're using for prediction, saving as "best"
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        print("the prediction we're making use of is " + str(best))
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# load and save model
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    # pass
    print(predicted[x], x_test[x], y_test[x])

# Drawing and plotting model
plot = "failures"
plt.scatter(df[plot], df["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()