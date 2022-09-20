
import unittest
import tensorflow as tf
from tfts.models.tcn import TCN


class TCNTest(unittest.TestCase):
    def test_model(self):
        custom_model_params = {}
        model = TCN(custom_model_params)

        x = tf.random.normal([16, 160, 36])
        y = model(x)
        self.assertEqual(y.shape, (16, ), 'incorrect output shape')

    def test_train(self):
        data_loader = DataLoader('sine')
        dataset = data_loader(params={}, data_dir=None, batch_size=8, training=True)
        print(dataset.take(1))

        inputs = tf.keras.layers.Input([30, 2])
        cnn_model = CNN()
        model = tf.keras.Model(inputs, cnn_model(inputs))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=3e-4))
        model.fit(dataset, epochs=5)


if __name__ == '__main__':
    unittest.main()
