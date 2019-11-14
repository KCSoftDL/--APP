import tensorflow as tf

class myscale(tf.keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(myscale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w1',
                                  shape=[1, 1, 1, 3],
                                  dtype=tf.float32,
                                  initializer='uniform',
                                  trainable=True)
        # Be sure to call this at the end
        super(myscale, self).build(input_shape)

        def call(self, inputs):
            return tf.matmul(inputs, self.w1)

        def compute_output_shape(self, input_shape):
            return tf.TensorShape(input_shape)

        def get_config(self):
            base_config = super(myscale, self).get_config()
            base_config['output_dim'] = self.output_dim
            return base_config

        @classmethod
        def from_config(cls, config):
            return cls(**config)