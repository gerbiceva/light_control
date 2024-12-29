from node import Node
from device_drivers import Gamepad
from data import Integer, Float, Vector2D


class GamepadNode(Node):
    def __init__(self, pad: Gamepad):
        super().__init__(self.get_game_state)
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
            "Down",
        ]

        for button in self.buttons:
            self.outputs[button] = Integer(self)

        self.outputs["R2"] = Float(self)
        self.outputs["L2"] = Float(self)

        self.outputs["Left stick"] = Vector2D(self)
        self.outputs["Right stick"] = Vector2D(self)

    def get_gamepad_state(self):
        return (self.pad.buttons[button] for button in self.buttons) + (
            self.pad.buttons["R2"],
            self.pad.buttons["L2"],
            self.pad.sticks["Left"],
            self.pad.sticks["Right"],
        )
