"""
Additional utilities expanded:
- MetricsAggregator for logging metrics counters and observations.
"""
import gc
import logging
from logger_config import log_metric

logger = logging.getLogger(__name__)

class MetricsAggregator:
    def __init__(self):
        self.counters = {}
        self.observations = {}
    
    def increment(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value
        log_metric(name, self.counters[name], kind="counter")
    
    def observe(self, name, value):
        arr = self.observations.get(name, [])
        arr.append(value)
        self.observations[name] = arr
        log_metric(name, value, kind="observation")
    
    def summary(self):
        return {
            'counters': self.counters,
            'observations': {k: {
                'count': len(v),
                'avg': sum(v)/len(v) if v else 0,
                'max': max(v) if v else 0,
                'min': min(v) if v else 0
            } for k,v in self.observations.items()}
        }

# (Existing FileManager, MemoryManager, ConfigManager preserved if needed)