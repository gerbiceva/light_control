import time

def seconds() -> float:
    return time.time()

class SpeedMaster:
    def __init__(self):
        self.speed = 1
        self.time = 0
    def increment(self):
        self.time = (self.time + 0.033333 * self.speed) % 1
    def speed_up(self):
        self.speed *= 2.0
    def slow_down(self):
        self.speed *= 0.5