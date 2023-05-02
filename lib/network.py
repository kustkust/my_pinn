import tensorflow as tf

class Network:
    """
    Build a physics informed neural network (PINN) model for Fick's equation.
    """

    @classmethod
    def build(cls, layers=[32, 16, 32], activation='tanh'):
        """

        Args:
            num_inputs: number of input variables. Default is 4 for (x, y, z, t).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default 5 for c(t, x, y, z), v_x(t, x, y, z), v_y(t, x, y, z), v_z(t, x, y, z),  Ф(t, x, y, z).

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(4,))
        # hidden layers
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)
        # output layer
        outputs_n = tf.keras.layers.Dense(1,
            kernel_initializer='he_normal')(x)
        outputs_v = tf.keras.layers.Dense(3,
            kernel_initializer='he_normal')(x)
        outputs_Fi = tf.keras.layers.Dense(1,
            kernel_initializer='he_normal')(x)

        tf.keras.models.Sequential

        return tf.keras.models.Model(inputs=inputs, outputs=[outputs_n, outputs_v, outputs_Fi])
