"""Test the U-net model"""

import unittest

import tensorflow as tf

from tfts.models.unet import Decoder, Encoder, Unet, UnetConfig


class UnetTest(unittest.TestCase):
    def test_encoder(self):
        """Test the encoder component of the UNet model."""
        # Test configuration
        units = 64
        kernel_size = 3
        depth = 1
        batch_size = 2
        seq_length = 16
        num_features = 3

        # Create encoder
        encoder = Encoder(
            units=units,
            kernel_size=kernel_size,
            depth=depth,
            use_attention=False,
            use_residual=False,
            use_se=False,
            use_layer_norm=False,
        )

        # Create test input
        x = tf.random.normal([batch_size, seq_length, num_features])
        pool1 = tf.random.normal([batch_size, seq_length // 2, num_features])
        pool2 = tf.random.normal([batch_size, seq_length // 4, num_features])

        # Test forward pass
        outputs = encoder([x, pool1, pool2], training=True)

        # Check outputs
        self.assertEqual(len(outputs), 4, "Encoder should return 4 outputs")
        self.assertEqual(outputs[0].shape, (batch_size, seq_length, units), "First output shape incorrect")
        self.assertEqual(outputs[1].shape, (batch_size, seq_length // 2, units * 2), "Second output shape incorrect")
        self.assertEqual(outputs[2].shape, (batch_size, seq_length // 4, units * 3), "Third output shape incorrect")
        self.assertEqual(outputs[3].shape, (batch_size, seq_length // 8, units * 4), "Fourth output shape incorrect")

    def test_decoder(self):
        """Test the decoder component of the UNet model."""
        # Test configuration
        units = 64
        kernel_size = 3
        predict_seq_length = 4
        batch_size = 2
        seq_length = 16

        # Create decoder
        decoder = Decoder(
            upsampling_factors=(2, 2, 2),
            units=units,
            kernel_size=kernel_size,
            predict_seq_length=predict_seq_length,
            use_attention=False,
            use_residual=False,
            use_se=False,
            use_layer_norm=False,
        )

        # Create test inputs
        out_0 = tf.random.normal([batch_size, seq_length, units])
        out_1 = tf.random.normal([batch_size, seq_length // 2, units * 2])
        out_2 = tf.random.normal([batch_size, seq_length // 4, units * 3])
        x = tf.random.normal([batch_size, seq_length // 8, units * 4])

        # Test forward pass
        output = decoder([out_0, out_1, out_2, x], training=True)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_length, units), "Decoder output shape incorrect")

    def test_model(self):
        """Test the complete UNet model."""
        # Test configuration
        predict_sequence_length = 4
        batch_size = 2
        seq_length = 16
        num_features = 3

        # Create model with default config
        model = Unet(predict_sequence_length=predict_sequence_length)
        x = tf.random.normal([batch_size, seq_length, num_features])
        y = model(x)
        self.assertEqual(y.shape, (batch_size, predict_sequence_length, 1), "incorrect output shape")

        # Test with custom config
        config = UnetConfig(
            units=32,
            kernel_size=3,
            depth=1,
            pool_sizes=(2, 4),
            upsampling_factors=(2, 2, 2),
            num_attention_heads=2,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            use_residual=False,
            use_attention=False,
            use_se=False,
            use_layer_norm=False,
        )
        model = Unet(predict_sequence_length=predict_sequence_length, config=config)
        y = model(x)
        self.assertEqual(y.shape, (batch_size, predict_sequence_length, 1), "incorrect output shape")

        # Test with dictionary output
        y_dict = model(x, return_dict=True)
        self.assertIsInstance(y_dict, dict, "should return dictionary when return_dict=True")
        self.assertIn("output", y_dict, "dictionary should contain 'output' key")
        self.assertEqual(
            y_dict["output"].shape, (batch_size, predict_sequence_length, 1), "incorrect output shape in dictionary"
        )

    def test_train(self):
        """Test the training functionality of the UNet model."""
        # Test configuration
        predict_sequence_length = 4
        batch_size = 2
        seq_length = 16
        num_features = 3

        # Create model
        backbone = Unet(predict_sequence_length=predict_sequence_length)
        inputs = tf.keras.layers.Input((seq_length, num_features))
        outputs = backbone(inputs)
        model = tf.keras.Model(inputs, outputs=outputs)

        # Create dummy data
        x = tf.random.normal([batch_size, seq_length, num_features])
        y = tf.random.normal([batch_size, predict_sequence_length, 1])

        # Create optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Test training step
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)

        # Check gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(any(g is not None for g in gradients), "No gradients were computed")

        # Test model compilation
        model.compile(optimizer=optimizer, loss=loss_fn)
        history = model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)
        self.assertIn("loss", history.history, "Training history should contain loss")
