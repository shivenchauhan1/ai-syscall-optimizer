"""
System Call Optimizer
Implements batching and caching strategies
"""

import time
from collections import defaultdict, OrderedDict

class SyscallOptimizer:
    def __init__(self, cache_size=100):
        """Initialize optimizer"""
        self.cache_size = cache_size
        self.cache = OrderedDict()  # LRU cache
        self.batch_buffer = defaultdict(list)
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batched_calls': 0,
            'total_calls': 0
        }
    
    def cache_get(self, key):
        """Get value from cache (LRU)"""
        self.stats['total_calls'] += 1
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats['cache_hits'] += 1
            return self.cache[key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def cache_put(self, key, value):
        """Put value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                # Remove least recently used
                self.cache.popitem(last=False)
    
    def add_to_batch(self, syscall_type, call_data):
        """Add syscall to batch buffer"""
        self.batch_buffer[syscall_type].append(call_data)
        self.stats['total_calls'] += 1
    
    def execute_batch(self, syscall_type):
        """Execute batched calls of same type"""
        if syscall_type in self.batch_buffer:
            batch = self.batch_buffer[syscall_type]
            count = len(batch)
            self.stats['batched_calls'] += count
            
            # Clear batch
            self.batch_buffer[syscall_type] = []
            
            return count
        return 0
    
    def get_cache_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        if total == 0:
            return 0
        return (self.stats['cache_hits'] / total) * 100
    
    def get_stats(self):
        """Get optimization statistics"""
        return {
            'cache_hit_rate': self.get_cache_hit_rate(),
            'total_calls': self.stats['total_calls'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'batched_calls': self.stats['batched_calls']
        }