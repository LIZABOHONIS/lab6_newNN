import math
import matplotlib.pyplot as plt


def exp(x):
    e = 2.71828182846
    return e ** x


def sigmoid(x):
    return 1 / (1 + exp(-x))


def multy_2_vectors(v1, v2):
    if len(v1) != len(v2):
        return None
    v3 = 0.0
    for i in range(len(v1)):
        v3 += v1[i] * v2[i]
    return v3


def plot_cost_function():
   plt.plot(dj0)
   plt.xlabel("item")
   plt.ylabel("J")
   plt.title("COST FUNCTION")
   plt.legend(['J = -(1/m) * (y * np.log(a) + (1 - y) * np.log(1 - a))'])
   plt.grid(True)
   plt.show()

x1 = [1, 1]
x2 = [1, 0]
x3 = [0, 1]
x4 = [0, 0]

x = [[1, 1, 0, 0],
     [1, 0, 1, 0]]

y = [1, 1, 1, 0]

w = [3.4, 3.4]
print('Initial weights : ', w)

b = 0
print('Initial vector : ', b)

m = 4
alpha = 0.1
eps = 0.00001
epoch = 10

z = [0, 0, 0, 0]
a = [0, 0, 0, 0]
dz = [0, 0, 0, 0]

dw1 = dw2 = 0
dw = [0, 0]

J0 = J1 = J2 = J3 = [0]
dj0 = dj1 = dj2 = dj3 = [0]

db = 0

for i in range(epoch):
    # суматор
    z[0] = multy_2_vectors(w, x1) + b
    z[1] = multy_2_vectors(w, x2) + b
    z[2] = multy_2_vectors(w, x3) + b
    z[3] = multy_2_vectors(w, x4) + b

    # активатор
    a[0] = sigmoid(z[0])
    a[1] = sigmoid(z[1])
    a[2] = sigmoid(z[2])
    a[3] = sigmoid(z[3])

    # cost function
    J0.append(-(y[0] * math.log(a[0]) + (1 - y[0]) * math.log(1 - a[0])))
    J1.append(-(y[1] * math.log(a[1]) + (1 - y[1]) * math.log(1 - a[1])))
    J2.append(-(y[2] * math.log(a[2]) + (1 - y[2]) * math.log(1 - a[2])))
    J3.append(-(y[3] * math.log(a[3]) + (1 - y[3]) * math.log(1 - a[3])))

    # похибка суматора
dz[1] = a[1] - y[1]
dz[2] = a[2] - y[2]
dz[3] = a[3] - y[3]

# похибка ваг
dw[0] += (1 / m) * (multy_2_vectors(x[0], dz))
dw[1] += (1 / m) * (multy_2_vectors(x[1], dz))

# похибка вектора зміщення
db += (1 / m) * (sum(dz))

# оновлені ваги
w[0] = w[0] - alpha * dw[0]
w[1] = w[1] - alpha * dw[1]

# оновлений вектор зміщення
b = b - alpha * db

dj0.append(((1 / m) * J0[i] - J0[i - 1]))
dj1.append(((1 / m) * J1[i] - J1[i - 1]))
dj2.append(((1 / m) * J2[i] - J2[i - 1]))
dj3.append(((1 / m) * J3[i] - J3[i - 1]))

dj = [dj0[-1], dj1[-1], dj2[-1], dj3[-1]]

for i in dj:
    if abs(i) <= eps:
        break

result = []
for i in a:
    if i > 0.5:
        result.append(1)
    else:
        result.append(0)
print('New Weights : ', w)
print('New vector : ', b)
print('a = ', a)
print('Result - ', result)
print('Cost function = ', dj)
plot_cost_function()

