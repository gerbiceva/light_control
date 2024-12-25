from typing import Callable

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
            results = self.f(*[d.data for d in self.inputs.values()])
            if isinstance(results, tuple):
                if len(self.outputs) != len(results):
                    print("Node function outputs len incorrect")
                for data_obj, calculated in zip(
                    self.outputs.values(), results
                ):
                    if calculated is not None:
                        data_obj.data = calculated
            elif len(self.outputs) != 0:
                next(iter(self.outputs.values())).data = results

            self.evaluated = True

class Data:
    def __init__(self, node: Node, initial_data=None):
        self.node = node
        self.data = initial_data
