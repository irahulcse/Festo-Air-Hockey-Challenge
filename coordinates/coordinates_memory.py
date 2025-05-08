import math
from collections import deque

class CoordinateBuffer:
    def __init__(self, max_length=50):
        self.buffer = deque(maxlen=max_length)

    def add(self, x : float, y : float):
        if self._is_valid(x) and self._is_valid(y):
            self.buffer.append((x, y))
        else:
            print("Invalid coordinate")

    def latest(self, anz):
        return list(self.buffer[-anz:]) if self.buffer else None

    def get_all(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    @staticmethod
    def _is_valid(value):
        return isinstance(value, (int, float)) and not math.isnan(value)