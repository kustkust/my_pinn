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
            Fi, Fi_grd, Fi_lap, j_div = \
            self.grads(ins[0])
        outs.append(c_t + j_div)
        outs.append(Fi_lap + 4*pi*l*kT*(z*c))
        outs.append(p_grd + (kT*c_grd + z*e*c*Fi_grd))
        # eq_4 = v_div

        c_l, Fi_l = self.network(ins[1])
        outs.append(c_l - c_left)
        # eq_v_left = v_l - v_left
        outs.append(Fi_l - Fi_left)

        c_r, Fi_r = self.network(ins[2])
        outs.append(c_r - c_right)
        # eq_v_right = v_r - v_right
        outs.append(Fi_r - Fi_right)

        c_s, Fi_s = self.network(ins[3])
        outs.append(c_s - c_start)
        # eq_v_start = v_s - v_start
        outs.append(Fi_s - ins[4])

        return tf.keras.models.Model(inputs=ins, outputs=outs)
