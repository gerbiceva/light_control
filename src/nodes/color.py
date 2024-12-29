from node import Node
from data import Float, ColorArray
import numpy as np

class SetBrightnessAllNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Color Array"] = ColorArray(self)
        self.inputs["Brightness"] = Float(self)
        self.outputs["Color Array"] = ColorArray(self)

    def change_number(self, hsv: np.array, value: float) -> ColorArray:
        new = hsv.copy()
        new[2, :] = value
        return new

class SetHueAllNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Color Array"] = ColorArray(self)
        self.inputs["Hue"] = Float(self)
        self.outputs["Color Array"] = ColorArray(self)

    def change_number(self, hsv: np.array, value: float) -> ColorArray:
        new = hsv.copy()
        new[0, :] = value
        return new