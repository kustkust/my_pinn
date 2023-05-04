import numpy as np

p: dict[str, float] = {
    "D": 0.006075,
    "q": 1.,
    "l": 0.7095,
    "kT": 1.,
    "nu": 79.53,
    "m": 1.,
    "Fi_left": -0.05,
    "Fi_right": -0.05,
    "c_left": 0.01,
    "c_right": 0.01,
    "width": 50.,
    "dim": 2,
    "max_dim": np.array([100., 100., 6., 6.]),
    "density_water": 26.15,
    "z": 1.,
    "e": 1.,
    "ro": 26.18,
}
p["max_dim"] = p["max_dim"][0:1+p["dim"]]
p["tdim"] = slice(0, 1)
p["sdim"] = slice(1, p["dim"]+1)
p["xi"] = p["D"]/p["kT"]