import matplotlib.pyplot as plt
import numpy as np


class SOM:
    def __init__(self, node_count=3, dimensions=4, eta_decay_rate=1e-2, initial_sigma=3.0,
                 sigma_decay_rate=1e-2, min_sigma=0.1,
                 initial_eta=0.1, min_eta=0.01):
        self.nodes = np.random.random_sample(size=(node_count, dimensions))
        self.node_count = node_count

        self.eta_decay_rate = eta_decay_rate
        self.initial_eta = initial_eta
        self.min_eta = min_eta

        self.sigma_decay_rate = sigma_decay_rate
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma

    def init_from_dataset(self, dataset):
        for i in range(len(self.nodes)):
            self.nodes[i] = dataset[np.random.randint(len(dataset))]

    @staticmethod
    def euclidean_distance(a, b):
        return (a - b) ** 2

    @staticmethod
    def decay(x, decay_rate, initial_value, min_value):
        return min_value + (initial_value - min_value) * (1 - decay_rate) ** x

    def gaussian_neighborhood(self, x, step):
        return np.exp(-((x ** 2) / (2 * self.sigma(step) ** 2)))

    def ricker_wavelet(self, x, step):
        a = self.sigma(step)
        y = (2 / (np.sqrt(3 * a) * np.power(np.pi, 1 / 3)))\
            * (1 - (x / a) ** 2) * np.exp(-(x ** 2) / (a ** 2))

        y[y > 1.0] = 1.0
        return y

    def predict(self, point):
        return SOM.euclidean_distance(self.nodes, point).sum(axis=1).argmin()

    def sigma(self, step):
        return SOM.decay(step, self.sigma_decay_rate, self.initial_sigma, self.min_sigma)

    def eta(self, step):
        return SOM.decay(step, self.eta_decay_rate, self.initial_eta, self.min_eta)

    def topological_neighborhood(self, distances, step):
        return self.ricker_wavelet(distances, step)

    def plot_topological_neighborhood(self, step=0):
        x = np.linspace(-10, 10, 1000)
        y = self.topological_neighborhood(x, step)
        plt.plot(x, y)
        plt.show()

    def fit(self, dataset, steps=1):
        for step in range(steps):
            sample = dataset[step % len(dataset)]
            winner = self.predict(sample)

            distance_from_winner = SOM.euclidean_distance(self.nodes, self.nodes[winner])
            topological_neighborhood = self.topological_neighborhood(distance_from_winner, step)

            self.nodes += self.eta(step) * topological_neighborhood * (sample - self.nodes)
