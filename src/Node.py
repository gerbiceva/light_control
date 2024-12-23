from typing import Callable

class Node:
    def __init__(self, f: Callable, input_types: list[str] = [],
                  output_types: list[str] = []):
        self.f = f
        self.outputs: list[Data]= []
        for output_type in output_types:
            self.outputs.append(Data(output_type))
        self.output_types = output_types

        self.inputs: list[Data] = []
        self.input_types = input_types

        self.evaluated = False

    def evaluate():
        if not self.evaluated:
            for input in self.inputs:
                self.inputs.node.evaluate()
            (data_obj.data = calculated for data_obj, calculated 
                                        in zip(self.outputs,
                                               self.f(*[d.data for d in self.inputs])))
            self.evaluated = True

class Data:
    def __init__(self, data_type, node, initial_data = None):
        self.data_type = data_type
        self.node = node
        self.data = initial_data
