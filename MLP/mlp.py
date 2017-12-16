import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score
from sklearn import neural_network


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)


class MLP():
    def __init__(self, sizes, input_size):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        print(len(self.biases),self.biases[0].shape)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            print(w.shape, a.shape, b.shape)
            a = sigmoid_vec(np.dot(w, a) + b)
        return a

    def GD(self, training_data, epochs, mini_batch_size, eta):
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        b = [np.zeros(b.shape) for b in self.biases]
        w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            b = [new_b + new_delta_b for new_b, new_delta_b in zip(b, delta_b)]
            w = [new_w + new_delta_w for new_w, new_delta_w in zip(w, delta_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, b)]

    def backprop(self, x, y):
        b = [np.zeros(b.shape) for b in self.biases]
        w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoid_prime_vec(zs[-1])
        b[-1] = delta
        w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * spv
            b[-l] = delta
            w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return b, w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

class MLPClassify(MLP):
    def __init__(self, input_size):
        MLP.__init__(self, input_size)

    def predict_prob(self, X):
        return self.feedforward(X)

    def predict(self, X):
        prob = self.predict_prob(X)
        prob[prob > 0] = 1
        prob[prob <= 0] = 0
        return prob


if __name__ == '__main__':
    X, y = datasets.load_breast_cancer(return_X_y=True)
    ds = [[i, j] for i, j in zip(datasets.load_breast_cancer()['data'], datasets.load_breast_cancer()['target'])]
    X = scale(X)
    mlp = MLP([2,3,1],30)

    sk_mlp = neural_network.MLPClassifier()
    sk_mlp.fit(X, y)
    print(roc_auc_score(y, mlp.evaluate(ds)))
    print(roc_auc_score(y, sk_mlp.predict(X)))
