import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def cost_func(the, x, y):
    m_size = y.size
    j = 1 / m_size * (-y.T.dot(np.log(sigmoid(x.dot(the)))) - (1 - y.T) * (np.log(1 - sigmoid(x.dot(the)))))
    grad = ((sigmoid(x.dot(the)) - y).T.dot(x.dot(1 / m_size))).T   
    return j, grad


def cost_func_reg(the, x, y, l):
    m_size = y.size
    j = 1 / m_size * (-y.T.dot(np.log(sigmoid(x.dot(the)))) - (1 - y.T) * (np.log(1 - sigmoid(x.dot(the))))) + 1 / (
        2 * m_size) * ((the ** 2).sum() - the[1] ** 2) * l
    grad = ((sigmoid(x.dot(the)) - y).T.dot(x.dot(1 / m_size))).T + theta * l / m_size
    grad['y'][0] -= l * the[0] / m_size
    return j, grad


def gradient_descent(x, y, the, alpha, num):
    m_size = y.size
    tmp = the
    for i in range(num):
        tmp = tmp - (x.T.dot(sigmoid(x.dot(the)) - y)) * alpha / m_size
    return tmp


def gradient_descent_reg(x, y, the, alpha, l, num):
    m_size = y.size
    tmp = the
    for i in range(num):
        tmp_add = alpha * 1 / m_size * l * the[0]
        tmp = tmp - alpha * 1 / m_size * ((x.T.dot(sigmoid(x.dot(the)) - y)) + l * the)
        tmp[0] += tmp_add
    return tmp


def newton_descent(x, y, the, l, num):
    m_size = y.size
    tmp = the
    for i in range(num):
        delta = 1 / (x.T.dot(sigmoid(x.dot(the)) - y)).dot(cost_func_reg(tmp, x, y, l)[0] * m_size)
        tmp = tmp - delta
        for k in delta:
            print(k < 0.01)
            if k < 0.02:
                return tmp
    return tmp


def map_feature(x1, x2):
    x1 = np.array(pd.DataFrame(x1))
    x2 = np.array(pd.DataFrame(x2))
    degree = 6
    res = np.ones((x1.size, 1))
    for i in range(degree):
        for j in range(i + 1):
            res = np.hstack((res, (x1 ** (i - j)) * (x2 ** j)))
    return res


def map_feature1(x1, x2):
    degree = 6
    res = np.ones((1, 1))
    for i in range(degree):
        for j in range(i + 1):
            res = np.hstack((res, np.array([(x1 ** (i - j)) * (x2 ** j)]).reshape(1, 1)))
    return res


def plot_decision(the, x):
    plt.plot(dataSet[dataSet.y == 0]['x1'], dataSet[dataSet.y == 0]['x2'], 'bx')
    plt.plot(dataSet[dataSet.y == 1]['x1'], dataSet[dataSet.y == 1]['x2'], 'ro')
    plot_x = [min(x[:, 2]) - 2, max(x[:, 2]) + 2]
    plot_y = (-1 / the[2]) * (the[1] * plot_x + the[0])
    plt.plot(plot_x, plot_y)
    plt.show()


def plot_decision_circle(new_theta):
    plt.plot(dataSet[dataSet.y == 0]['x1'], dataSet[dataSet.y == 0]['x2'], 'bx')
    plt.plot(dataSet[dataSet.y == 1]['x1'], dataSet[dataSet.y == 1]['x2'], 'ro')
    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = sigmoid((map_feature1(u[i], v[j])).dot(new_theta))
    print(z)
    plt.contour(u, v, z, 8)
    plt.show()


dataSet = pd.DataFrame(pd.read_table('ex2data2.txt', sep=','))
# print(dataSet)
# plt.plot(dataSet[dataSet.y==0]['x1'],dataSet[dataSet.y==0]['x2'],'bx')
# plt.plot(dataSet[dataSet.y==1]['x1'],dataSet[dataSet.y==1]['x2'],'ro')
# plt.show()

X = dataSet.loc[:, ['x1', 'x2']]
Y = dataSet.loc[:, ['y']]
m, n = X.shape

# # regular
X = map_feature(X['x1'], X['x2'])
lamb = 1
# X = np.hstack((np.ones((m, 1)), X))

initial_theta = np.zeros((X.shape[1], 1))
# cost, grad = cost_func_reg(initial_theta, X, y, lamb)
# print(cost,grad)

# cost, grad = cost_func(test_theta, X, y)
# print(cost, grad)
theta = gradient_descent_reg(X, Y, initial_theta, 0.2, 0.001, 1000)
# theta = newtonDescent(X, y, initial_theta, 0.01, 500)
plot_decision_circle(theta)
# cost, grad = cost_func(theta, X, y)
# print(cost, grad)
# plotDecision(reg_theta,X,y)
