from collections import deque

class CoordinateBuffer:
    def __init__(self, max_length=50):
        self.buffer = deque(maxlen=max_length)

    def add(self, x, y):
        self.buffer.append((x, y))

    def get_all(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

