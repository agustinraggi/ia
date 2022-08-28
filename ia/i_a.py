import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles 

# crear el dataset

n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c="skyblue")
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c="salmon")
plt.axis("equal")
plt.show()

# clase de la de la red

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.acr_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 -1
        self.w = np.random.rand(n_conn, n_neur) * 2 -1

# funcion de activacion 
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 -x))

_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))