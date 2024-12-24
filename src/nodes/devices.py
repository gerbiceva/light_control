from ..node import Node
from ..device_drivers import Gamepad
from ..data import Integer, Float, Vector2D

class GamepadNode(Node):
    def __init__(self, pad: Gamepad):
        super().__init__(self.get_game_state())
        self.buttons = [
            "A",
            "B",
            "X",
            "Y",
            "L1",
            "R1",
            "Start",
            "Select",
            "L3",
            "R3",
            "Left",
            "Right",
            "Up",
            "Down"
        ]

        for button in self.buttons:
            self.outputs[button] = Integer()

        self.outputs["R2"] = Float()
        self.outputs["L2"] = Float()

        self.outputs["Left stick"] = Vector2D()
        self.outputs["Right stick"] = Vector2D()


    def get_gamepad_state(self):
        for button in self.buttons:
            self.outputs[button].data = self.pad.buttons[button]

        self.outputs["R2"] = self.pad.buttons["R2"]
        self.outputs["L2"] = self.pad.buttons["L2"]

        self.outputs["Left stick"] = self.pad.sticks["Left"]
        self.outputs["Right stick"] = self.pad.sticks["Right"]
