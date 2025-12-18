"""
Simple caching layer (in-memory + optional filesystem).
"""
import json
import time
from pathlib import Path
from threading import RLock

class CacheManager:
    def __init__(self, enabled=True, cache_dir="cache", ttl=3600):
        self.enabled = enabled
        self.ttl = ttl
        self.lock = RLock()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory = {}
    
    def _is_expired(self, entry):
        return time.time() - entry['time'] > self.ttl
    
    def get(self, key):
        if not self.enabled:
            return None
        with self.lock:
            entry = self.memory.get(key)
            if entry and not self._is_expired(entry):
                return entry['value']
            if entry:
                del self.memory[key]
            return None
    
    def set(self, key, value):
        if not self.enabled:
            return
        with self.lock:
            self.memory[key] = {'value': value, 'time': time.time()}