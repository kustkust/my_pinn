import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import math
import tensorflow as tf


def train(pinn, p, epochs=600):
    n_train_samples = 10

    max_d = np.array([p["max_t"], p["max_x"], p["max_y"], p["max_z"]])

    x_train = np.random.rand(n_train_samples, 4) * max_d
    x_train_left = np.random.rand(n_train_samples, 4) * max_d
    x_train_left[:,1] = 0
    x_train_right = np.random.rand(n_train_samples, 4) * max_d
    x_train_right[:,1] = 0
    x=[x_train, x_train_left, x_train_right]

    y = [np.zeros((n_train_samples, 1))] * 11
    
    pinn.fit(x=x, y=y, epochs=epochs)
    # pinn.save_weights('weights.h5')


def predict(network, params):
    num_test_samples = 10
    t_flat = np.linspace(0, 1, num_test_samples)
    x_flat = np.linspace(0, 1, num_test_samples)
    y_flat = np.linspace(0, 1, num_test_samples)
    z_flat = np.linspace(0, 1, num_test_samples)
    t, x, y, z = np.meshgrid(t_flat, x_flat, y_flat, z_flat)
    txyz = np.stack([t.flatten(), x.flatten(),
                    y.flatten(), z.flatten()], axis=-1)
    n, v, Fi = network.predict(txyz, batch_size=num_test_samples)

    return n, v, Fi


def main(epochs=600):
    # number of test samples
    num_test_samples = 10

    # system parameters
    p: dict[str, float] = {
        "D": 0.006075,
        "W": 0.,
        "q": 1.,
        "l": 0.7095,
        "kT": 1.,
        "eta": 79.53,
        "m": 1.,
        "Fi_left": 1,
        "Fi_right": 1.,
        "n_left": 1.,
        "n_right": 1.,
        "width": 50.,
        "max_t": 1.,
        "max_x": 100.,
        "max_y": 6.,
        "max_z": 6.,
        "density_water": 26.15,
    }
    p["nu"] = p["D"]/p["kT"]
    p["eta_B"] = p["eta"]/p["density_water"]

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, m=p["m"]).build(**p)
    pinn.compile(optimizer='adam', loss='mse')

    train(pinn, p)
    n, v, Fi = predict(pinn, p)


if __name__ == "__main__":
    main(epochs=600)
