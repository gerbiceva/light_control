import time
from datatypes import node, Float, initialize, Int, each_tick

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
        self.speed *= 1.1
    def slow_down(self):
        self.speed *= 0.9

speedmasters = []
@each_tick
def tick():
    for master in speedmasters:
        master.increment()

@node
@initialize
def make_speedmaster():
    """
    Speedmaster

    Gives you a MASTER of SPEEEEED(zan oberstar).

    Parameters
    ----------
    speed_up : Float
    speed_down : Float

    Returns
    -------
    time : Float
    """
    speed_master = SpeedMaster()
    speedmasters.append(speed_master)
    def get_speed_master(speed_up: Int, speed_down: Int) -> Float:
        if speed_up:
            speed_master.speed_up()
        if speed_down:
            speed_master.slow_down()
        return speed_master.time
    return get_speed_master

@node
@initialize
def manual_speed():
    """
    Manual Speedmaster

    Gives you a MASTER of SPEEEEED(zan oberstar).

    Parameters
    ----------
    delta : Float

    Returns
    -------
    time : Float
    """
    now = 0.0
    def speed_master(delta):
        nonlocal now
        now += delta
        return now
    
    return speed_master
    