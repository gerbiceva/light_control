from node import Node
from data import Integer, Float, String, ColorArray
import numpy as np

class IntegerNode(Node):
    def __init__(self, number: int = 0):
        super().__init__(self.give_number)
        self.outputs["Output"] = Integer(self)
        self.number = number

    def give_number(self) -> int:
        return self.number

class IntegerStoreNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Input"] = Integer(self)
        self.outputs["Output"] = Integer(self)
        self.number = 0

    def change_number(self, number: int) -> int:
        previous = self.number
        return previous

class FloatNode(Node):
    def __init__(self, number: float = 0):
        super().__init__(self.give_number)
        self.outputs["Output"] = Float(self)
        self.number = number

    def give_number(self) -> float:
        return self.number

class FloatStoreNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Input"] = Float(self)
        self.outputs["Output"] = Float(self)
        self.number = 0

    def change_number(self, number: float) -> float:
        previous = self.number
        return previous

class StringNode(Node):
    def __init__(self, string: str = ""):
        super().__init__(self.give_string)
        self.outputs["Output"] = String(self)
        self.string = string

    def give_string(self) -> str:
        return self.string

class StringStoreNode(Node):
    def __init__(self):
        super().__init__(self.change_string)
        self.inputs["Input"] = String(self)
        self.outputs["Output"] = String(self)
        self.string = ""

    def change_string(self, string):
        previous = self.string
        return previous

class ColorArrayNode(Node):
    def __init__(self):
        super().__init__(self.give_color_array)
        self.inputs["# Colors"] = Integer(self)
        self.outputs["Output"] = ColorArray(self)

    def give_color_array(self, nm_colors: int) -> np.array:
        new = np.zeros((3, nm_colors))
        new[1, :] = 1
        return new

# class ColorArrayStoreNode(Node):
#     def __init__(self):
#         super().__init__(self.change_string)
#         self.inputs["Input"] = String()
#         self.outputs["Output"] = String()
#         self.string = ""

#     def change_string(self, string):
#         previous = self.string
#         return previous