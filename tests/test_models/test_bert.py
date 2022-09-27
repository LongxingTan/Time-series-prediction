
import unittest
import tensorflow as tf
from tfts.models.bert import Bert


class BertModelTester:
    def __init__(self):
        pass

    def prepare_config_and_inputs(self):
        return

    def prepare_config_and_inputs_for_decoder(self):
        return

    def create_and_check_model(self):
        return

    def create_and_check_model_as_decoder(self):
        return


class BertModelTest(unittest.TestCase):
    all_model_classes = (

    )

    def setup(self):
        self.model_tester = BertModelTester()
        self.config_tester = ConfigTester()

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)


class BertModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        pass
