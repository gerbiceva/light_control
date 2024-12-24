from node import Node
import numpy as np

class Data:
    def __init__(self, node: Node, initial_data=None):
        self.node = node
        self.data = initial_data

class ColorArray(Data):
    def __init__(self, node: Node, initial_data=None):
        super().__init__(node, initial_data)

class Vector2D(Data):
    def __init__(self, node: Node, initial_data=np.array([0,0])):
        super().__init__(node, initial_data)

class Integer(Data):
    def __init__(self, node: Node, initial_data: int = 0):
        super().__init__(node, initial_data)

class Float(Data):
    def __init__(self, node: Node, initial_data: float = 0):
        super().__init__(node, initial_data)

class String(Data):
    def __init__(self, node: Node, initial_data: str = ""):
        super().__init__(node, initial_data)