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
        txyz = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        txyz_left = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        txyz_right = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        txyz_start = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        Fi_start =  tf.keras.layers.Input(1)

        c, c_t, c_grd, \
            v, v_t, v_div, v_adv, v_lap, \
            Fi, Fi_grd, Fi_lap, j_div = \
            self.grads(txyz)
        eq_1 = c_t + j_div
        eq_2 = Fi_lap + 4*pi*l*kT*(z*c)
        eq_3 = (v_t + v_adv)*ro + p_grd - nu*v_lap + (kT*c_grd + z*e*c*Fi_grd)
        eq_4 = v_div

        c_l, c_t_l, c_grd_l, \
            v_l, v_t_l, v_div_l, v_adv_l, v_lap_l, \
            Fi_l, Fi_grd_l, Fi_lap_l, j_div_l = \
            self.grads(txyz_left)
        eq_c_left = c_l - c_left
        eq_v_left = v_l - v_left
        eq_Fi_left = Fi_l - Fi_left

        c_r, c_t_r, c_grd_r, \
            v_r, v_t_r, v_div_r, v_adv_r, v_lap_r, \
            Fi_r, Fi_grd_r, Fi_lap_r, j_div_r = \
            self.grads(txyz_right)
        eq_c_right = c_r - c_right
        eq_v_right = v_r - v_right
        eq_Fi_right = Fi_r - Fi_right

        c_s, c_t_s, c_grd_s, \
            v_s, v_t_s, v_div_s, v_adv_s, v_lap_s, \
            Fi_s, Fi_grd_s, Fi_lap_s, j_div_s = \
            self.grads(txyz_start)
        eq_c_start = c_s - c_start
        eq_v_start = v_s - v_start
        eq_Fi_start = Fi_s - Fi_start

        return tf.keras.models.Model(
            inputs=[txyz, txyz_left, txyz_right, txyz_start, Fi_start],
            outputs=[eq_1, eq_2, eq_3, eq_4,
                     eq_c_left, eq_v_left, eq_Fi_left,
                     eq_c_right, eq_v_right, eq_Fi_right,
                     eq_c_start, eq_v_start, eq_Fi_start,])

        
        # return tf.keras.models.Model(
        #     inputs=[txyz, txyz_left, txyz_right, txyz_start],
        #     outputs={"eq1": eq_1, "eq2": eq_2, "eq3": eq_3, "eq4": eq_4,
        #              "cl": eq_c_left, "vl": eq_v_left, "Fil": eq_Fi_left,
        #              "cr": eq_c_right, "vr": eq_v_right, "Fir": eq_Fi_right,
        #              "cs": eq_c_start, "vs": eq_v_start, "Fis": eq_Fi_start, })
