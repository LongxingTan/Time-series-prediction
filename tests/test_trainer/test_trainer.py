
import unittest
from tfts.trainer import Trainer


class TrainerTest(unittest.TestCase):
    def setUp(self):
        batch_size = 128

    def tearDown(self):
        pass

    def test_trainer_no_dist_strategy(self):
        pass

    def test_trainer_static_batch(self):
        pass

    def test_trainer_1gpu_with_dist_strategy(self):
        pass

    def test_trainer_2gpu(self):
        pass

    def test_trainer_fp16(self):
        pass

    def test_trainer_2gpu_fp16(self):
        pass

    def test_predict(self):
        pass

    def test_predict_fp16(self):
        pass


class KerasTrainerTest(unittest.TestCase):
    def setUp(self):
        pass
