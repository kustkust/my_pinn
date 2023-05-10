import tensorflow as tf


class Network:
    @classmethod
    def build(cls, layers=[32, 16, 32], activation='tanh', **kwargs):

        # input layer
        inputs = tf.keras.layers.Input(shape=(1+kwargs["dim"],))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                                      kernel_initializer='he_normal')(x)
        # output layer
        outputs_n = tf.keras.layers.Dense(1,
                                          kernel_initializer='he_normal')(x)
        # outputs_v = tf.keras.layers.Dense(kwargs["dim"],
        #                                   kernel_initializer='he_normal')(x)
        outputs_Fi = tf.keras.layers.Dense(1,
                                           kernel_initializer='he_normal')(x)

        tf.keras.models.Sequential

        return tf.keras.models.Model(inputs=inputs, outputs=[outputs_n, outputs_Fi])
