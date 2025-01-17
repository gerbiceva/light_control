from typing import Callable
from inspect import signature
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
        # print(self._parameters, self._returns)
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
        self.missing = len(self._parameters)
        self.used = False

class Graph:
    def __init__(self):
        self.nodes: dict[int: Callable] = {}
        self.edges: dict[tuple[int, int]: list[tuple[int, int]]] = {}
        self.params: dict[ParametersQueue] = {}
    
    def construct(self, nodes: dict[int: Callable], edges: dict[tuple[int, int]: list[tuple[int, int]]]):
        self.nodes: dict[int, Callable] = nodes
        self.edges: dict[tuple[int, int]: list[tuple[int, int]]] = edges
        self.params: dict[int, ParametersQueue] = {id: ParametersQueue(node) for id, node in self.nodes.items()}

    def evaluate(self):
        while not all([param.used for param in self.params.values()]):
            # print([f"{id}: {param.missing}" for id, param in self.params.items()])
            stuck = True
            for i in self.nodes.keys():
                param = self.params[i]
                node = self.nodes[i]
                if param.missing == 0 and not param.used:
                    # print(f"executed {i}")
                    results = node(*param.get_parameters())
                    # print(i)
                    # print(results)
                    stuck = False
                    if results is None:
                        break
                    if not isinstance(results, tuple):
                        results = (results,)
                    for j, result in zip(param.get_returns(), results):
                        # print("aaaaaaaaaaaaa")
                        # print(self.edges)
                        # print((i, j))
                        # print()
                        for edge in self.edges[(i, j)]:
                            self.params[edge[0]].set_parameter(edge[1], result)
            if stuck:
                # print("nothing to do")
                break

        for param in self.params.values():
            param.clear()
