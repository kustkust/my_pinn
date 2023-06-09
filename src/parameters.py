import numpy as np
import tensorflow as tf


def build_params(dim):
    p = {
        "D": 0.006075,
        "q": 1.,
        "l": 0.7095,
        "kT": 1.,
        "nu": 79.53,
        "m": 1.,
        "c_left": 0.01, "c_right": 0.01, "c_start": 0.002,
        "v_left": 0., "v_right": 0., "v_start": 0,
        "Fi_left": -0.05, "Fi_right": -0.05, "Fi_start": np.array([[0,-0.05]]),
        "width": 50.,
        "dim": dim,
        "density_water": 26.15,
        "z": 1.,
        "e": 1.,
        "ro": 26.15,
        "p_grd": tf.constant([0.,1.,0.]),
    }
    trim = slice(0, 1+p["dim"])
    if dim == 1:
        p["v_const"] = np.array([1.0])
    else :
        p["v_const"] = np.array([0., 1., 0.])[0:dim]
    p["Fi_start"] = np.loadtxt("Fi.dat")
    p["p_grd"] = p["p_grd"][0:dim]
    p["min_dim"] = np.array([ 0., -25., -3., -3.])[trim]
    p["max_dim"] = np.array([10.,  25.,  3.,  3.])[trim]
    p["size"] = p["max_dim"] - p["min_dim"]
    p["tdim"] = slice(0, 1)
    p["sdim"] = slice(1, 1+p["dim"])
    p["xi"] = p["D"]/p["kT"]
    return p