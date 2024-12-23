from input import Gamepad
from light import Light
import time
import sacn
import numpy as np
from utils import FrameLimiter
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

sender = sacn.sACNsender(bind_address="192.168.0.101")
sender.start()
sender.manual_flush = True
light = Light(7, sender, 100)

pad = Gamepad("/dev/input/event8")

limiter = FrameLimiter(30)

while True:
    strip[0, :] = np.atan2(*pad.sticks["Right"]) / (2 * np.pi) + 0.5
    strip[1, :] = 1 - pad.buttons["L2"]
    strip[2, :] = pad.buttons["R2"] / 12
    print(strip[:, :3])
    limiter.tick()  # Ensures the function runs 30 times per second
    sender.flush()
