from pycrdt import Array, Text, TypedDoc, TypedMap


class Ydata(TypedMap):
    capability: Text
    value: Text


class Ymeasuerd(TypedMap):
    width: int
    height: int


class Ypos(TypedMap):
    x: int
    y: int


class YNode(TypedMap):
    id: int
    type: Text
    position: Ypos
    measured: Ymeasuerd
    data: Ydata


class YGraph(TypedMap):
    id: int
    name: Text
    description: Text
    nodes: YNode[bool]
    edges: Array[bool]


class MyDoc(TypedDoc):
    graph: YGraph


doc = MyDoc()

doc.data.name = "foo"
doc.data.toggle = False
# doc.data.toggle = 3  # error: Incompatible types in assignment (expression has type "int", variable has type "bool")  [assignment]
doc.array0 = Array([1, 2, 3])
# doc.data.nested = Array(
#     [4]
# )  # error: List item 0 has incompatible type "int"; expected "bool"  [list-item]
doc.data.nested = Array([False, True])
# v0: str = doc.data.name
# v1: str = doc.data.toggle  # error: Incompatible types in assignment (expression has type "bool", variable has type "str")  [assignment]
# v2: bool = doc.data.toggle
# doc.data.wrong_key0  # error: "MyMap" has no attribute "wrong_key0"  [attr-defined]
