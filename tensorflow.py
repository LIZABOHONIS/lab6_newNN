from sklearn.neural_network import MLPRegressor
import numpy as np

X = [[1, 1], [1, 0], [0, 1], [0, 0]]
Y= [1, 1, 1, 0]

model = MLPRegressor(hidden_layer_sizes=(2),
                     activation='logistic',
                     solver='lbfgs')
model.fit(X,Y)

test_X = [[1, 1], [1, 0], [0, 1], [0, 0]]
for i in range(len(X)):
  print(f"x_test[ {i+1} ] = {test_X[i]}")
  test = np.array(test_X[i]).reshape(1, -1)

  predict = model.predict(test)
  result = predict > 0.5 if True else False

  print(f"predict = {predict}")
  print(f"result = {result}\n")
