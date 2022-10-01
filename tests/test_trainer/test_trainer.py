import unittest

from absl import flags

from tfts.trainer import KerasTrainer, Trainer

FLAGS = flags.FLAGS


# class TrainerTest(unittest.TestCase):
#     def setUp(self):
#         FLAGS.n_epochs = 10
#         batch_size = 32
#         learning_rate = 0.0003
#         model_dir = ""
#
#     def tearDown(self):
#         pass
#
#     def test_trainer_basic(self):
#         # 1gpu, no dist, no
#         pass
#
#     def test_trainer_no_dist_strategy(self):
#         pass
#
#     def test_trainer_static_batch(self):
#         pass
#
#     def test_trainer_1gpu_with_dist_strategy(self):
#         pass
#
#     def test_trainer_2gpu(self):
#         pass
#
#     def test_trainer_fp16(self):
#         pass
#
#     def test_trainer_2gpu_fp16(self):
#         pass
#
#     def test_predict(self):
#         pass
#
#     def test_predict_fp16(self):
#         pass
#
#
# class KerasTrainerTest(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
