import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class Pipeline(object):
    _load_processor = False

    def __init__(self, model, processor: Optional):
        pass

    def __call__(self):
        pass

    def forward(self):
        pass

    def predict(self):
        pass

    def get_iterator(self):
        pass

    def postprocess(self):
        pass
