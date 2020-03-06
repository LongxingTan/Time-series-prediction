
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense


class GAN(object):
    def __init__(self):
        self.generator=Generator()
        self.discriminator=Discriminator()

    def __call__(self, inputs_shape):
        x=Input(inputs_shape)
        encoder_output,encoder_state=self.encoder(x)
        decoder_output = self.decoder(None,encoder_state,x)
        return tf.keras.Model(x,decoder_output)


class Generator(tf.keras.Model):
    def __init__(self,params):
        super(Generator,self).__init__()
        self.params=params
        self.upconv1=tf.keras.layers.Conv2DTranspose(filters=64,
                                                     kernel_size=[4,1],
                                                     strides=[2,1],
                                                     padding='SAME')
        self.upconv2 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                       kernel_size=[4, 1],
                                                       strides=[2, 1],
                                                       padding='SAME')
        self.upconv3=tf.keras.layers.Conv2DTranspose(filters=2,
                                                     kernel_size=[4,1],
                                                     strides=[2,1],
                                                     padding='SAME')
        self.fc1=tf.keras.layers.Dense(units=1024)
        self.fc2=tf.keras.layers.Dense(units=4*1*128)
        self.bn1=tf.keras.layers.BatchNormalization()
        self.bn2=tf.keras.layers.BatchNormalization()
        self.bn3=tf.keras.layers.BatchNormalization()

    def call(self,z,training=True):
        ln1 = tf.nn.relu(self.bn1(self.fc1(z)))
        ln2 = tf.nn.relu(self.bn2(self.fc2(ln1)))
        ln2 = tf.reshape(ln2, [-1, 4, 1, 128])
        print('ln2', ln2.get_shape().as_list())

        upconv1 = tf.nn.relu(self.bn3(self.upconv1(ln2)))

        print('upconv1', upconv1.get_shape().as_list())
        upconv2 = tf.nn.relu(self.upconv2(upconv1))
        output = tf.nn.sigmoid(self.upconv3(upconv2))
        print('generator output shape', output.get_shape().as_list())
        return output


class Discriminator(tf.keras.Model):
    def __init__(self,params):
        super(Discriminator,self).__init__()
        self.params=params
        self.conv1=tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding='SAME')
        self.conv2=tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=[4,4],
                                          strides=[2,2],
                                          padding='SAME')
        self.fc1=tf.keras.layers.Dense(1024)
        self.fc2=tf.keras.layers.Dense(1)
        self.bn1=tf.keras.layers.BatchNormalization()
        self.bn2=tf.keras.layers.BatchNormalization()

    def call(self,x,training=True):
        x = tf.convert_to_tensor(x)  # class Tensor has dtype of float64_ref and class Variable has dtype of float64
        conv1 = tf.nn.leaky_relu(self.conv1(x))
        conv2 = tf.nn.leaky_relu(self.bn1((conv1)))
        conv2 = tf.reshape(conv2, [self.params['batch_size'], -1])
        ln1 = tf.nn.leaky_relu(self.bn2(self.fc1(conv2)))
        ln2 = self.fc2(ln1)
        output = tf.nn.sigmoid(ln2)
        return output
