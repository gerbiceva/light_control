from typing import Callable
import numpy as np

class Node:
    def __init__(self, f: Callable):
        self.f = f
        self.evaluated = False
        self.inputs: dict[Data] = {}
        self.outputs: dict[Data] = {}

    def evaluate(self) -> None:
        if not self.evaluated:
            for input in self.inputs.values():
                if input.node != self:
                    input.node.evaluate()
            
            if any([input.fresh for input in self.inputs.values()]) or (len(self.inputs) == 0):
                results = self.f(*[d.data for d in self.inputs.values()])

                # If node has multiple outputs
                if isinstance(results, tuple):
                    if len(self.outputs) != len(results):
                        print("Node function outputs len incorrect")
                    for data_obj, result in zip(
                        self.outputs.values(), results
                    ):
                        if data_obj.data == result:
                            data_obj.fresh = False
                        else:
                            data_obj.data = result
                            data_obj.fresh = True
                # If node has one output
                elif len(self.outputs) != 0:
                    data_obj = next(iter(self.outputs.values()))
                    if type(data_obj.data) is not np.ndarray:
                        if data_obj.data == results:
                            data_obj.fresh = False
                        else:
                            data_obj.data = results
                            data_obj.fresh = True
                    else:
                        if np.array_equal(data_obj.data, results):
                            data_obj.fresh = False
                        else:
                            data_obj.data = results
                            data_obj.fresh = True

            self.evaluated = True

class Data:
    def __init__(self, node: Node, initial_data=None):
        self.node = node
        self.data = initial_data
        self.fresh = True
