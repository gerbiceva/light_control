from node import Node
from data import ColorArray, Integer
import numpy as np
from matplotlib.colors import hsv_to_rgb
from sacn import sACNsender


class StripNode(Node):
    def __init__(self, sender: sACNsender):
        super().__init__(self.send_data)
        self.inputs["Universe"] = Integer(self)
        self.inputs["LED Colors"] = ColorArray(self)
        self.inputs["Universe"].data = 1
        self.universe = 1
        self.sender: sACNsender = sender
        self.sender.activate_output(self.universe)
        self.sender[self.universe].multicast = True

    def send_data(self, universe: int, hsv: np.array):
        if universe != self.universe:
            self.sender.deactivate_output(self.universe)
            self.universe = universe
            self.sender.activate_output(universe)
            self.sender[universe].multicast = True

        self.sender[self.universe].dmx_data = np.pad(
            (hsv_to_rgb(hsv.T).flatten() * 255).astype(np.uint8),
            (0, 512 - hsv.shape[0] * hsv.shape[1]),
            mode="constant",
            constant_values=0,
        ).tobytes()
        print(f"sent data {hsv}")