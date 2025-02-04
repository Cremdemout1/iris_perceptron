from perceptron import Perceptron
from io import StringIO
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

response = requests.get(s)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text), header=None,  encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# # plot data
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='x', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='o', label='versicolor')
# plt.xlabel('sepal length cm')
# plt.ylabel('petal length cm')
# plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(learning_rate=0.1, n_iter=6)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()
def loop():
    print("Please write 'exit' to leave and 'examine' to enter AI Iris predictor")
    while (1):
        user_input = input()
        if user_input == "exit":
            break
        elif user_input == "examine":
            try:
                sepalLength = float(input("Enter sepal length of Iris: "))  # Convert input to float
                petalLength = float(input("Enter petal length of Iris: "))  # Convert input to float
            except ValueError:
                print("Please enter valid numerical values.")
                continue
            iris = "Iris Setosa" if ppn.predict([sepalLength, petalLength]) == -1 else "Iris Versicolor"
            print(iris)
loop()