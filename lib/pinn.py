import tensorflow as tf
from math import pi
from .layer import GradientLayer


class PINN:

    def __init__(self, network, **kwargs):
        self.network = network
        self.grads = GradientLayer(self.network, **kwargs)

    def build(self, nu, z, l, kT, ro, e, Fi_left, Fi_right, c_left, c_right, **kwargs):
        txyz = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        txyz_left = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        txyz_right = tf.keras.layers.Input(shape=(1+kwargs["dim"],))

        c, c_t, c_grd, v, v_t, v_div, v_adv, v_lap, Fi, Fi_grd, Fi_lap, j_div = self.grads(txyz)
        eq_1 = c_t + j_div
        eq_2 = Fi_lap + 4*pi*l*kT*(z*c)
        eq_3 = (v_t + v_adv)*ro - nu*v_lap + (kT*c_grd + z*e*c*Fi_grd)
        eq_4 = v_div

        c, c_t, c_grd, v, v_t, v_div, v_adv, v_lap, Fi, Fi_grd, Fi_lap, j_div = self.grads(txyz_left)
        eq_n_left = c - c_left
        v_left = v
        eq_Fi_left = Fi - Fi_left

        c, c_t, c_grd, v, v_t, v_div, v_adv, v_lap, Fi, Fi_grd, Fi_lap, j_div = self.grads(txyz_right)
        eq_n_right = c - c_right
        v_right = v
        eq_Fi_right = Fi - Fi_right

        return tf.keras.models.Model(
            inputs=[txyz, txyz_left, txyz_right], 
            outputs=[
                eq_1, eq_2, eq_3, eq_4,  
                eq_n_left, v_left, eq_Fi_left, 
                eq_n_right, v_right, eq_Fi_right])
