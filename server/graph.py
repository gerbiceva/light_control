from typing import Callable
from numpydoc.docscrape import FunctionDoc

class ParametersQueue:
    def __init__(self, func: Callable):
        self._parameters = {}
        self._returns = []
        for par in FunctionDoc(func)["Parameters"]:
            if par.name != "None":
                self._parameters[par.name] = None
        for ret in FunctionDoc(func)["Returns"]:
            if ret.name != "":
                self._returns.append(ret.name)
        self._required = func.__required__ if hasattr(func, '__required__') else None
        if self._required:
            self.missing = self._required
        else:
            self.missing = len(self._parameters)
        self.used = False

    def get_parameters(self):
        self.used = True
        return self._parameters.values()

    def get_returns(self):
        return self._returns

    def set_parameter(self, index, value):
        self._parameters[index] = value
        self.missing -= 1

    def clear(self):
        for key in self._parameters.keys():
            self._parameters[key] = None
        if self._required:
            self.missing = self._required
        else:
            self.missing = len(self._parameters)
        self.used = False

class Graph:
    def __init__(self, nodes: dict[int: Callable], edges: dict[tuple[int, int]: list[tuple[int, int]]]):
        self.nodes: dict[int, Callable] = nodes
        self.edges: dict[tuple[int, int]: list[tuple[int, int]]] = edges
        self.params: dict[int, ParametersQueue] = {id: ParametersQueue(node) for id, node in self.nodes.items()}

    def evaluate(self):
        while not all([param.used for param in self.params.values()]):
            stuck = True
            for i in self.nodes.keys():
                param = self.params[i]
                node = self.nodes[i]
                if param.missing == 0 and not param.used:
                    results = node(*param.get_parameters())
                    stuck = False
                    if results is None:
                        break
                    if not isinstance(results, tuple):
                        results = (results,)
                    for j, result in zip(param.get_returns(), results):
                        for edge in self.edges[(i, j)]:
                            if result is None:
                                continue
                            self.params[edge[0]].set_parameter(edge[1], result)
            if stuck:
                break

        for param in self.params.values():
            param.clear()
