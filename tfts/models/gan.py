"""
`Generative Adversarial Networks
<https://arxiv.org/abs/1406.2661>`_
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Input

params = {
    "rnn_size": 32,
    "dense_size": 8,
    "num_stacked_layers": 1,
    "predict_window_sizes": 5,
}


class GAN(object):
    """GAN model"""

    def __init__(self, custom_model_params):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def __call__(self, inputs_shape, training):
        x = Input(inputs_shape)
        generator_output = self.generator(x)
        decoder_output = self.discriminator(generator_output, x)
        return tf.keras.Model(x, decoder_output)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.upconv1 = Conv2DTranspose(filters=64, kernel_size=[4, 1], strides=[2, 1], padding="SAME")
        self.upconv2 = Conv2DTranspose(filters=32, kernel_size=[4, 1], strides=[2, 1], padding="SAME")
        self.upconv3 = Conv2DTranspose(filters=2, kernel_size=[4, 1], strides=[2, 1], padding="SAME")
        self.fc1 = Dense(units=1024)
        self.fc2 = Dense(units=4 * 1 * 128)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, z, training=True):
        """_summary_

        Parameters
        ----------
        z : _type_
            _description_
        training : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        ln1 = tf.nn.relu(self.bn1(self.fc1(z)))
        ln2 = tf.nn.relu(self.bn2(self.fc2(ln1)))
        ln2 = tf.reshape(ln2, [-1, 4, 1, 128])
        print("ln2", ln2.get_shape().as_list())

        upconv1 = tf.nn.relu(self.bn3(self.upconv1(ln2)))

        print("upconv1", upconv1.get_shape().as_list())
        upconv2 = tf.nn.relu(self.upconv2(upconv1))
        output = tf.nn.sigmoid(self.upconv3(upconv2))
        print("generator output shape", output.get_shape().as_list())
        return output


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
        self.conv2 = Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
        self.fc1 = Dense(1024)
        self.fc2 = Dense(1)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, x, training=True):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        training : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        x = tf.convert_to_tensor(x)  # class Tensor has dtype of float64_ref and class Variable has dtype of float64
        conv1 = tf.nn.leaky_relu(self.conv1(x))
        conv2 = tf.nn.leaky_relu(self.bn1((conv1)))
        conv2 = tf.reshape(conv2, [x.get_shape().as_list()[0], -1])
        ln1 = tf.nn.leaky_relu(self.bn2(self.fc1(conv2)))
        ln2 = self.fc2(ln1)
        output = tf.nn.sigmoid(ln2)
        return output
