import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def costFunc(theta, X, y):
    m = y.size
    J = 1 / m * (-y.T.dot(np.log(sigmoid(X.dot(theta)))) - (1 - y.T).dot(np.log(1 - sigmoid(X.dot(theta)))))
    grad = ((sigmoid(X.dot(theta)) - y).T.dot(X.dot(1 / m))).T
    return (J, grad)


def costFuncReg(theta, X, y, lamb):
    m = y.size
    J = 1 / m * (-y.T.dot(np.log(sigmoid(X.dot(theta)))) - (1 - y.T).dot(np.log(1 - sigmoid(X.dot(theta))))) + 1 / (
        2 * m) * ((theta ** 2).sum() - theta[1] ** 2) * lamb
    grad = ((sigmoid(X.dot(theta)) - y).T.dot(X.dot(1 / m))).T + theta * lamb / m
    grad['y'][0] -= lamb * theta[0] / m
    return (J, grad)


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    tmp = theta
    for i in range(num_iters):
        tmp = tmp - (X.T.dot(sigmoid(X.dot(theta)) - y)) * alpha / m
    return tmp


def gradientDescentReg(X, y, theta, alpha, lamb, num_iters):
    m = y.size
    tmp = theta
    for i in range(num_iters):
        tmp_add = alpha * 1 / m * lamb * theta[0]
        tmp = tmp - alpha * 1 / m * ((X.T.dot(sigmoid(X.dot(theta)) - y)) + lamb * theta)
        tmp[0] += tmp_add
    return tmp


def newtonDescent(X, y, theta, lamb, num_iters):
    m = y.size
    tmp = theta
    for i in range(num_iters):
        delta = 1 / (X.T.dot(sigmoid(X.dot(theta)) - y)).dot(costFuncReg(tmp, X, y, lamb)[0] * m)
        tmp = tmp - delta
        for k in delta:
            print(k < 0.01)
            if k < 0.02:
                return tmp
    return tmp


def mapFeature(x1, x2):
    x1 = np.array(pd.DataFrame(x1))
    x2 = np.array(pd.DataFrame(x2))
    degree = 6
    res = np.ones((x1.size, 1))
    for i in range(degree):
        for j in range(i + 1):
            res = np.hstack((res, (x1 ** (i - j)) * (x2 ** j)))
    return res


def mapFeature1(x1, x2):
    degree = 6
    res = np.ones((1, 1))
    for i in range(degree):
        for j in range(i + 1):
            res = np.hstack((res, np.array([(x1 ** (i - j)) * (x2 ** j)]).reshape(1,1)))
    return res


def plotDecision(theta, X, y):
    plt.plot(dataSet[dataSet.y == 0]['x1'], dataSet[dataSet.y == 0]['x2'], 'bx')
    plt.plot(dataSet[dataSet.y == 1]['x1'], dataSet[dataSet.y == 1]['x2'], 'ro')
    plot_x = [min(X[:, 2]) - 2, max(X[:, 2]) + 2]
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y)
    plt.show()


def plotDecisionCircle(new_theta, X, y):
    plt.plot(dataSet[dataSet.y == 0]['x1'], dataSet[dataSet.y == 0]['x2'], 'bx')
    plt.plot(dataSet[dataSet.y == 1]['x1'], dataSet[dataSet.y == 1]['x2'], 'ro')
    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = sigmoid((mapFeature1(u[i],v[j])).dot(new_theta))
    print(z)
    plt.contour(u,v,z,8)
    plt.show()


dataSet = pd.DataFrame(pd.read_table('ex2data2.txt', sep=','))
# print(dataSet)
# plt.plot(dataSet[dataSet.y==0]['x1'],dataSet[dataSet.y==0]['x2'],'bx')
# plt.plot(dataSet[dataSet.y==1]['x1'],dataSet[dataSet.y==1]['x2'],'ro')
# plt.show()

X = dataSet.loc[:, ['x1', 'x2']]
y = dataSet.loc[:, ['y']]
m, n = X.shape

# # regular
X = mapFeature(X['x1'], X['x2'])
lamb = 1
#X = np.hstack((np.ones((m, 1)), X))

initial_theta = np.zeros((X.shape[1], 1))
#cost, grad = costFuncReg(initial_theta, X, y, lamb)
#print(cost,grad)

# cost, grad = costFunc(test_theta, X, y)
# print(cost, grad)
theta = gradientDescentReg(X, y, initial_theta, 0.2, 0.001, 1000)
#theta = newtonDescent(X, y, initial_theta, 0.01, 500)
plotDecisionCircle(theta, X, y)
#cost, grad = costFunc(theta, X, y)
#print(cost, grad)
# reg_theta=np.array([[1.273005],[0.624876],[1.177376],[-2.020142],[-0.912616],[-1.429907],[0.125668],[-0.368551],[-0.360033],[-0.171068],[-1.460894],[-0.052499],[-0.618889],[-0.273745],[-1.192301],[-0.240993],[-0.207934],[-0.047224],[-0.278327],[-0.296602],[-0.453957],[-1.045511]])

# plotDecision(reg_theta,X,y)
