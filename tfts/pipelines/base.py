import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Pipeline(object):
    _load_processor = False

    def __init__(self, task: str, model: str, processor: Optional[Callable] = None):
        self.task = task
        self.model = model
        self.processor = processor

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
