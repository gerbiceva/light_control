import time
from datatypes import node, Float, initialize

@node
@initialize
def make_seconds():
    """
    Seconds

    Gives seconds.

    Returns
    -------
    seconds : Float
    """
    now = time.time()
    def seconds() -> Float:
        return time.time() - now
    return seconds

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

speed_master = SpeedMaster()

def get_synced_time() -> float:
    return speed_master.time