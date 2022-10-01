import json
import unittest

import requests
import tensorflow as tf


class ServingTest(unittest.TestCase):
    @unittest.skip("only test while serving")
    def test_request(self):
        data = tf.random.normal([])
        headers = {"content-type": "application/json"}
        json_response = requests.post("http://localhost:8501/v1/models/tfts_model:predict", data=data, headers=headers)
        predictions = json.loads(json_response.text)
        print(predictions)
