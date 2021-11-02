from collections import defaultdict
import time

class Timer:
    timers = defaultdict(float)
    counters = defaultdict(int)

    def __init__(self, field_name=None):
        self.field_name = field_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        end = time.time()
        self.duration = end - self.start
        self.timers[self.field_name] += self.duration
        self.counters[self.field_name] += 1

    @classmethod
    def print_stats(cls):
        print('field_name', 'total_time', 'total_count', 'it/s', sep=', ')
        for field_name in cls.timers.keys():
            total_time = cls.timers[field_name]
            total_count = cls.counters[field_name]
            print(field_name, total_time, total_count, total_count / total_time, sep=', ')
