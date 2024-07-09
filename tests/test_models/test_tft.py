"""Test the TFT model"""

# class TFTransformerTest(unittest.TestCase):
#     def test_model(self):
#         predict_sequence_length = 8
#         custom_model_config = {}
#         model = TFTransformer(predict_sequence_length, custom_model_config=custom_model_config)
#
#         x = tf.random.normal([2, 16, 3])
#         y = model(x)
#         self.assertEqual(y.shape, (2, predict_sequence_length, 1), "incorrect output shape")
