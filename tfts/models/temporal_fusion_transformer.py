"""
`Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
<https://arxiv.org/abs/1912.09363>`_
"""

import tensorflow as tf

params = {
    "skip_connect_circle": False,
    "skip_connect_mean": False,
}


class TFTransformer(object):
    """Temporal fusion transformer model"""

    def __init__(self, predict_sequence_length=3, custom_model_params=None):
        if custom_model_params:
            params.update(custom_model_params)
        self.params = params
        self.predict_sequence_length = predict_sequence_length

    def __call__(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        return
