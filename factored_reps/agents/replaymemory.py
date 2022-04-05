from collections import defaultdict
import json
import os

import numpy as np

class ReplayMemory:
    def __init__(self, capacity: int, on_retrieve: dict = None):
        self.capacity = capacity
        self.on_retrieve = defaultdict(lambda: (lambda items: items))
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
        self.fields = self.fields.union(set(experience.keys()))
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, fields=None):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        return self.retrieve(idx, fields)

    def retrieve(self, idx=None, fields: list = None):
        if idx is None:
            idx = range(len(self.memory))
        try:
            experiences = [self.memory[i] for i in idx]
        except TypeError:
            experiences = [self.memory[idx]]

        if fields is None:
            fields = sorted(list(self.fields))
        if isinstance(fields, str):
            fields = [fields]

        result = []
        for field_name in fields:
            result.append(self._extract_array(experiences, field_name))
        result = tuple(result)
        if len(fields) == 1:
            result = result[0]

        return result

    def __len__(self):
        return len(self.memory)

    def _extract_array(self, experiences, key):
        items = [experience[key] for experience in experiences]
        if self.on_retrieve:
            if key != '_index_':
                items = self.on_retrieve['*'](items)
            items = self.on_retrieve[key](items)
        return items

    def save(self, directory, filename='replaymemory', extension='.json'):
        if extension != '.json':
            raise NotImplementedError
        fields = sorted(list(self.fields))

        data = [{key: serialize_np(value)
                 for key, value in experience.items()} for experience in self.memory]
        archive = {
            'capacity': self.capacity,
            'position': self.position,
            'fields': fields,
            'memory': data,
        }
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename + extension)
        with open(filepath, 'w') as fp:
            json.dump(archive, fp)

    def load(self, filepath):
        with open(filepath, 'r') as fp:
            archive = json.load(fp)
        self.capacity = int(archive['capacity'])
        self.position = int(archive['position'])
        self.fields = set(archive['fields'])
        self.memory = [{
            key: unserialize_np(value, dtype)
            for (key, (value, dtype)) in experience.items()
        } for experience in archive['memory']]

def serialize_np(value):
    output = value
    dtype = None
    try:
        output = value.tolist()
        dtype = str(value.dtype)
    except AttributeError:
        pass
    return output, dtype

def unserialize_np(value, dtype):
    output = value
    if dtype is not None:
        output = np.asarray(value, dtype=dtype)
    return output
