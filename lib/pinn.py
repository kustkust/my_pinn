import tensorflow as tf
from math import pi
from .layer import GradientLayer


class PINN:

    def __init__(self, network, m):
        self.network = network
        self.grads = GradientLayer(self.network, m)

    def build(self, D, nu, W, q, l, kT, eta, eta_b, Fi_left, Fi_right, n_left, n_right, **kwargs):
        txyz = tf.keras.layers.Input(shape=(4,))
        txyz_left = tf.keras.layers.Input(shape=(4,))
        txyz_right = tf.keras.layers.Input(shape=(4,))

        n, n_t, n_grd, j, j_div, Fi, Fi_grd, Fi_lap, v, v_t, v_adv, v_lap, v_div_grd, ro, ro_t, ro_grd, rov_div = self.grads(txyz)
        eq_1 = n_t + j_div
        eq_2 = j + D * n_grd + nu*q*n*Fi_grd - n*v # + tf.sqrt(n)*W
        eq_3 = Fi_lap + 4*pi*l*kT*(q*n)
        eq_4 = (v_t + v_adv)*ro + kT*ro_grd + \
            q*n*Fi_grd - eta*v_lap + (eta/3+eta_b)*v_div_grd
        eq_5 = ro_t + rov_div

        n, n_t, n_grd, j, j_div, Fi, Fi_grd, Fi_lap, v, v_t, v_adv, v_lap, v_div_grd, ro, ro_t, ro_grd, rov_div = self.grads(txyz_left)
        eq_n_left = n - n_left
        v_left = v
        eq_Fi_left = Fi - Fi_left

        n, n_t, n_grd, j, j_div, Fi, Fi_grd, Fi_lap, v, v_t, v_adv, v_lap, v_div_grd, ro, ro_t, ro_grd, rov_div = self.grads(txyz_right)
        eq_n_right = n - n_right
        v_right = v
        eq_Fi_right = Fi - Fi_right

        return tf.keras.models.Model(
            inputs=[txyz, txyz_left, txyz_right], 
            outputs=[
                eq_1, eq_2, eq_3, eq_4, eq_5, 
                eq_n_left, v_left, eq_Fi_left, 
                eq_n_right, v_right, eq_Fi_right])
