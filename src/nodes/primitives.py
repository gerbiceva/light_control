from ..nodes import Node
from ..data import Integer, Float, String

class IntegerNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Input"] = Integer()
        self.outputs["Output"] = Integer()
        self.number = 0

    def change_number(self, number):
        previous = self.number
        return previous

class FloatNode(Node):
    def __init__(self):
        super().__init__(self.change_number)
        self.inputs["Input"] = Float()
        self.outputs["Output"] = Float()
        self.number = 0

    def change_number(self, number):
        previous = self.number
        return previous

class StringNode(Node):
    def __init__(self):
        super().__init__(self.change_string)
        self.inputs["Input"] = String()
        self.outputs["Output"] = String()
        self.string = ""

    def change_string(self, string):
        previous = self.string
        return previous