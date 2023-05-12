import tensorflow as tf
import numpy as np
from math import pi
from .layer import GradientLayer


class PINN:

    def __init__(self, network, **kwargs):
        self.network = network
        self.grads = GradientLayer(self.network, **kwargs)

    def build(self, nu, z, l, kT, ro, e, p_grd,
              Fi_left, Fi_right,
              c_left, c_right, c_start,
              v_left, v_right, v_start,
              **kwargs):
        ins = []
        ins.append(tf.keras.layers.Input(shape=(1+kwargs["dim"],)))  # inside
        ins.append(tf.keras.layers.Input(shape=(1+kwargs["dim"],)))  # left
        ins.append(tf.keras.layers.Input(shape=(1+kwargs["dim"],)))  # right
        ins.append(tf.keras.layers.Input(shape=(1+kwargs["dim"],)))  # start
        ins.append(tf.keras.layers.Input(1))                         # Fi_start

        outs = []

        c, c_t, c_grd, \
            v_t, v_div, v_adv, v_lap, \
            Fi_grd, Fi_lap, j_div = \
            self.grads(ins[0])
        outs.append(c_t + j_div)
        outs.append(Fi_lap + 4*pi*l*kT*(z*c))
        outs.append(ro*(v_t + v_adv) + p_grd - nu*v_lap + (kT*c_grd + z*e*c*Fi_grd))
        outs.append(v_div)

        r_l = self.network(ins[1])
        c_l, v_l, Fi_l = r_l["c"], r_l["v"], r_l["Fi"]
        outs.append(c_l - c_left)
        outs.append(v_l - v_left)
        outs.append(Fi_l - Fi_left)

        r_r = self.network(ins[2])
        c_r, v_r, Fi_r = r_r["c"], r_r["v"], r_r["Fi"]
        outs.append(c_r - c_right)
        outs.append(v_r - v_right)
        outs.append(Fi_r - Fi_right)

        r_s = self.network(ins[2])
        c_s, v_s, Fi_s = r_s["c"], r_s["v"], r_s["Fi"]
        outs.append(c_s - c_start)
        outs.append(v_s - v_start)
        outs.append(Fi_s - ins[4])

        return tf.keras.models.Model(inputs=ins, outputs=outs)
