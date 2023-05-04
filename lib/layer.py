import tensorflow as tf
from utilities import print_t



class GradientLayerOld(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.p = kwargs
        super().__init__()
        # super().__init__(**kwargs)

    def call(self, input):
        with tf.GradientTape(persistent=True) as g:
            g.watch(input)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(input)
                n, v, Fi = self.model(input)
                ro = self.p["m"] * n
                rov = ro * v

            # spatial coords
            sdim = self.p["sdim"]

            print("n", n.shape)
            n_jac = gg.batch_jacobian(n, input)
            print("n_jac", n_jac.shape)
            n_t = n_jac[..., 0]
            print("n_t", n_t.shape)
            n_grd = n_jac[..., 0, sdim]
            print("n_grd", n_grd.shape)

            print("j", j.shape)
            j_grd = gg.batch_jacobian(j, input)[..., sdim]
            print("j_grd", j_grd.shape)
            j_div = tf.linalg.trace(j_grd)
            print("j_div", j_div.shape)

            print("Fi", Fi.shape)
            Fi_jac = gg.batch_jacobian(Fi, input)
            print("Fi_jac", Fi_jac.shape)
            Fi_grd = Fi_jac[..., 0, sdim]
            print("Fi_grd", Fi_grd.shape)

            print("v", v.shape)
            v_jac = gg.batch_jacobian(v, input)
            print("v_jac", v_jac.shape)
            v_t = v_jac[..., 0]
            print("v_t", v_t.shape)
            v_adv = tf.reduce_sum(tf.expand_dims(
                v, axis=1)*v_jac[..., sdim], axis=-2)
            print("v_adv", v_adv.shape)
            v_grd = v_jac[..., sdim]
            print("v_grd", v_grd.shape)
            v_div = tf.reduce_sum(v_grd, axis=-1)
            print("v_div", v_div.shape)

            print("ro", ro.shape)
            ro_jac = gg.batch_jacobian(ro, input)
            print("ro_jac", ro_jac.shape)
            ro_t = ro_jac[..., 0]
            print("ro_t", ro_t.shape)
            ro_grd = ro_jac[..., 0, sdim]
            print("ro_grd", ro_grd.shape)

            print("rov", rov.shape)
            rov_jac = gg.batch_jacobian(rov, input)
            print("rov_jac", rov_jac.shape)
            rov_div = tf.linalg.trace(rov_jac[..., sdim])
            print("rov_div", rov_div.shape)

        Fi_jac2 = g.batch_jacobian(Fi_grd, input)
        print("Fi_jac2", Fi_jac2.shape)
        Fi_lap = tf.linalg.trace(Fi_jac2[..., sdim])
        print("Fi_lap", Fi_lap.shape)

        v_div_jac = g.batch_jacobian(v_div, input)
        print("v_div_jac", v_div_jac.shape)
        v_div_grd = v_div_jac[..., 1:4]
        print("v_div_grd", v_div_grd.shape)
        # v_jac2 = tf.linalg.diag(g.batch_jacobian(v_jac, input))
        # v_lap = tf.reduce_sum(v_jac2[..., 1:4], axis=-1)
        v_lap = tf.linalg.trace(g.batch_jacobian(v_grd, input)[..., sdim])
        print("v_lap", v_lap.shape)

        return n, n_t, n_grd, j, j_div, Fi, Fi_grd, Fi_lap, v, v_t, v_adv, v_lap, v_div_grd, ro, ro_t, ro_grd, rov_div


class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.p = kwargs
        super().__init__()
        # super().__init__(**kwargs)

    def call(self, x):
        p = self.p
        if "print_debug" in p:
            print_debug = p["print_debug"]
        else:
            print_debug = False
        sdim = p["sdim"]
        tdim = p["tdim"]

        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                c, v, Fi = self.model(x)

            print_t(c, debug=print_debug)
            c_jac = g.batch_jacobian(c, x)[..., 0, :]
            print_t(c_jac, debug=print_debug)
            c_t = c_jac[..., tdim]
            print_t(c_t, debug=print_debug)
            c_grd = c_jac[..., sdim]
            print_t(c_grd, debug=print_debug)

            print_t(v, debug=print_debug)
            v_jac = g.batch_jacobian(v, x)
            print_t(v_jac, debug=print_debug)
            v_t = v_jac[..., tdim][..., 0]
            print_t(v_t, debug=print_debug)
            v_grd = v_jac[..., sdim]
            print_t(v_grd, debug=print_debug)
            v_div = tf.linalg.trace(v_grd)[:, None]
            print_t(v_div, debug=print_debug)
            v_adv = tf.reduce_sum(v[:, None]*v_grd, axis=-1)
            print_t(v_adv, debug=print_debug)

            print_t(Fi, debug=print_debug)
            Fi_jac = g.batch_jacobian(Fi, x)[..., 0, :]
            print_t(Fi_jac, debug=print_debug)
            Fi_grd = Fi_jac[..., sdim]
            print_t(Fi_grd, debug=print_debug)

            j = -p["D"]*c_grd - p["xi"]*p["z"]*p["e"]*c*Fi_grd + c*v
            print_t(j, debug=print_debug)

        v_grd2 = gg.batch_jacobian(v_grd, x)[..., sdim]
        print_t(v_grd2, debug=print_debug)
        v_lap = tf.linalg.trace(v_grd2)
        print_t(v_lap, debug=print_debug)

        Fi_grd_jac = gg.batch_jacobian(Fi_grd, x)
        print_t(Fi_grd_jac, debug=print_debug)
        Fi_lap = tf.linalg.trace(Fi_grd_jac[..., sdim])[:, None]
        print_t(Fi_lap, debug=print_debug)

        j_jac = gg.batch_jacobian(j, x)
        print_t(j_jac, debug=print_debug)
        j_div = tf.expand_dims(tf.linalg.trace(j_jac[..., sdim]), axis=1)
        print_t(j_div, debug=print_debug)

        return c, c_t, c_grd, v, v_t, v_div, v_adv, v_lap, Fi, Fi_grd, Fi_lap, j_div
