import tensorflow as tf


class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, model, m, **kwargs):
        self.model = model
        self.mass = m
        super().__init__(**kwargs)

    def call(self, input):
        with tf.GradientTape(persistent=True) as g:
            g.watch(input)
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(input)
                n, v, Fi = self.model(input)
                j = n * v
                ro = self.mass * n
                rov = ro * v

            n_jac = gg.batch_jacobian(n, input)
            n_t = n_jac[..., 0]
            n_grd = n_jac[..., 1:4]

            j_grd = gg.batch_jacobian(j, input)[..., 1:4]
            j_div = tf.reduce_sum(j_grd, axis=-1)

            Fi_jac = gg.batch_jacobian(Fi, input)
            Fi_grd = Fi_jac[..., 1:4]

            v_jac = gg.batch_jacobian(n, input)
            v_t = v_jac[..., 0]
            v_adv = tf.reduce_sum(tf.expand_dims(
                v, axis=1)*v_jac[..., 1:4], axis=-2)
            v_grd = v_jac[..., 1:4]
            v_div = tf.reduce_sum(v_grd, axis=-1)

            ro_jac = gg.batch_jacobian(ro, input)
            ro_t = ro_jac[..., 0]
            ro_grd = ro_jac[..., 1:4]

            rov_jac = gg.batch_jacobian(rov, input)
            rov_div = tf.reduce_sum(rov_jac[..., 1:4], axis=-1)

        Fi_jac2 = g.batch_jacobian(Fi_jac, input)
        Fi_lap = tf.reduce_sum(Fi_jac2[..., 1:4], axis=-1)

        v_div_jac = g.batch_jacobian(v_div, input)
        v_div_grd = v_div_jac[..., 1:4]
        # v_jac2 = tf.linalg.diag(g.batch_jacobian(v_jac, input))
        # v_lap = tf.reduce_sum(v_jac2[..., 1:4], axis=-1)
        v_lap = tf.linalg.trace(g.batch_jacobian(v_jac, input)[..., 1:4])

        return n, n_t, n_grd, j, j_div, Fi, Fi_grd, Fi_lap, v, v_t, v_adv, v_lap, v_div_grd, ro, ro_t, ro_grd, rov_div
