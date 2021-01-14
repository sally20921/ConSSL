import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engine import Events

class StatMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(StatMetric, self).__init__(output_transform)

        self.log_iter = True


