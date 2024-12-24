from typing import Callable
from data import Data


class Node:
    def __init__(self, f: Callable):
        self.f = f
        self.evaluated = False
        self.inputs: dict[Data] = {}
        self.outputs: dict[Data] = {}

    def evaluate(self):
        if not self.evaluated:
            for input in self.inputs.items():
                self.inputs.node.evaluate()
            for data_obj, calculated in zip(
                self.outputs.items(), self.f(*[d.data for d in self.inputs.items()])
            ):
                data_obj.data = calculated

            self.evaluated = True
