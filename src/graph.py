from typing import Callable
from inspect import signature

class ParametersQueue:
    def __init__(self, func: Callable):
        self._parameters = {}
        params = list(signature(func).parameters.values())
        for i, par in enumerate(params):
            self._parameters[i]: par.annotation = None
        self.missing = len(self._parameters)
        self.used = False

    def get_parameters(self):
        self.used = True
        return self._parameters.values()

    def set_parameter(self, index, value):
        self._parameters[index] = value
        self.missing -= 1

    def clear(self):
        for key in self._parameters.keys():
            self._parameters[key] = None
        self.missing = len(self._parameters)
        self.used = False

class Graph:
    def __init__(self, nodes: list[Callable], edges: dict[tuple[int, int]: list[tuple[int, int]]]):
        self.nodes: list[Callable] = nodes
        self.edges: dict[tuple[int, int]: list[tuple[int, int]]] = edges
        self.params: list[ParametersQueue] = [ParametersQueue(node) for node in self.nodes]

    def evaluate(self):
        while not all([param.used for param in self.params]):
            for i, (param, node) in enumerate(zip(self.params, self.nodes)):
                if param.missing == 0 and not param.used:
                    results = node(*param.get_parameters())
                    if results is None:
                        break
                    if not isinstance(results, tuple):
                        results = (results,)
                    for j, result in enumerate(results):
                        for edge in self.edges[(i, j)]:
                            self.params[edge[0]].set_parameter(edge[1], result)

        for param in self.params:
            param.clear()
