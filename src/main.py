import sacn
import numpy as np
from utils import FrameLimiter
from nodes.lights import StripNode
from nodes.primitives import FloatNode, ColorArrayNode, IntegerNode
from nodes.color import SetBrightnessAllNode

sender = sacn.sACNsender(bind_address="192.168.1.88")
sender.start()
sender.manual_flush = True

limiter = FrameLimiter(30)

# Example node construction
nodes = []
strip = StripNode(sender)
nodes.append(strip)

bright = SetBrightnessAllNode()
strip.inputs["LED Colors"] = bright.outputs["Color Array"]
nodes.append(bright)

start = ColorArrayNode()
bright.inputs["Color Array"] = start.outputs["Output"]
nodes.append(start)

nm_leds = IntegerNode(5)
start.inputs["# Colors"] = nm_leds.outputs["Output"]

level = FloatNode(0.5)
bright.inputs["Brightness"] = level.outputs["Output"]

strip.evaluate()





# while True:
    # strip[0, :] = np.atan2(*pad.sticks["Right"]) / (2 * np.pi) + 0.5
    # strip[1, :] = 1 - pad.buttons["L2"]
    # strip[2, :] = pad.buttons["R2"] / 12
    # limiter.tick()
    # sender.flush()
