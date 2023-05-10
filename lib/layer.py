import tensorflow as tf
from utilities import print_t


class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.p = kwargs
        super().__init__()
        # super().__init__(**kwargs)
        self.D = tf.constant(self.p["D"])
        self.xi = tf.constant(self.p["xi"])
        self.z = tf.constant(self.p["z"])
        self.e = tf.constant(self.p["e"])

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
                c, Fi = self.model(x)

            print_t(c, debug=print_debug)
            c_jac = g.batch_jacobian(c, x)[..., 0, :]
            print_t(c_jac, debug=print_debug)
            c_t = c_jac[..., tdim]
            print_t(c_t, debug=print_debug)
            c_grd = c_jac[..., sdim]
            print_t(c_grd, debug=print_debug)

            # print_t(v, debug=print_debug)
            # v_jac = g.batch_jacobian(v, x)
            # print_t(v_jac, debug=print_debug)
            # v_t = v_jac[..., tdim][..., 0]
            # print_t(v_t, debug=print_debug)
            # v_grd = v_jac[..., sdim]
            # print_t(v_grd, debug=print_debug)
            # v_div = tf.linalg.trace(v_grd)[:, None]
            # print_t(v_div, debug=print_debug)
            # v_adv = tf.reduce_sum(v[:, None]*v_grd, axis=-1)
            # print_t(v_adv, debug=print_debug)

            print_t(Fi, debug=print_debug)
            Fi_jac = g.batch_jacobian(Fi, x)[..., 0, :]
            print_t(Fi_jac, debug=print_debug)
            Fi_grd = Fi_jac[..., sdim]
            print_t(Fi_grd, debug=print_debug)

            j = -p["D"]*c_grd - p["xi"]*p["z"]*p["e"]*c*Fi_grd + c*p["v_const"]
            print_t(j, debug=print_debug)

        # v_grd2 = gg.batch_jacobian(v_grd, x)[..., sdim]
        # print_t(v_grd2, debug=print_debug)
        # v_lap = tf.linalg.trace(v_grd2)
        # print_t(v_lap, debug=print_debug)

        Fi_grd_jac = gg.batch_jacobian(Fi_grd, x)
        print_t(Fi_grd_jac, debug=print_debug)
        Fi_lap = tf.linalg.trace(Fi_grd_jac[..., sdim])[:, None]
        print_t(Fi_lap, debug=print_debug)

        j_jac = gg.batch_jacobian(j, x)
        print_t(j_jac, debug=print_debug)
        j_div = tf.expand_dims(tf.linalg.trace(j_jac[..., sdim]), axis=1)
        print_t(j_div, debug=print_debug)

        # return c, c_t, c_grd, v, v_t, v_div, v_adv, v_lap, Fi, Fi_grd, Fi_lap, j_div
        return c, c_t, c_grd, Fi, Fi_grd, Fi_lap, j_div
