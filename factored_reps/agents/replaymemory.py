from collections import defaultdict
import random

import numpy as np

class ReplayMemory:
    def __init__(self, capacity: int, on_retrieve: dict = None):
        self.capacity = capacity
        self.on_retrieve = defaultdict(lambda items: items)
        if on_retrieve is not None:
            self.on_retrieve.update(on_retrieve)

        self.fields = set()
        self.reset()

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        experience['_index_'] = self.position
        self.fields += set(experience.keys())
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, fields=None):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        return self.retrieve(idx, fields)

    def retrieve(self, idx=None, fields: list = None):
        if idx is None:
            idx = range(len(self.memory))
        experiences = [self.memory[i] for i in idx]

        if fields is None:
            fields = sorted(list(self.fields))

        result = []
        for field_name in fields:
            result += self._extract_array(experiences, field_name)
        result = tuple(result)

        return result

    def __len__(self):
        return len(self.memory)

    def _extract_array(self, experiences, key):
        items = [experience[key] for experience in experiences]
        if self.on_retrieve:
            items = self.on_retrieve['*'](items)
            items = self.on_retrieve[key](items)
        return items
