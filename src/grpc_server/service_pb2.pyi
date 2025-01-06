from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BaseType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Int: _ClassVar[BaseType]
    Float: _ClassVar[BaseType]
    String: _ClassVar[BaseType]
    Color: _ClassVar[BaseType]
    Curve: _ClassVar[BaseType]
    ColorArray: _ClassVar[BaseType]
    Array: _ClassVar[BaseType]
    Vector2D: _ClassVar[BaseType]
    Vector3D: _ClassVar[BaseType]
Int: BaseType
Float: BaseType
String: BaseType
Color: BaseType
Curve: BaseType
ColorArray: BaseType
Array: BaseType
Vector2D: BaseType
Vector3D: BaseType

class Void(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Port(_message.Message):
    __slots__ = ("name", "type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: BaseType
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[BaseType, str]] = ...) -> None: ...

class NodeCapability(_message.Message):
    __slots__ = ("name", "description", "inputs", "outputs")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    inputs: _containers.RepeatedCompositeFieldContainer[Port]
    outputs: _containers.RepeatedCompositeFieldContainer[Port]
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ...) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeCapability]
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeCapability, _Mapping]]] = ...) -> None: ...

class Edge(_message.Message):
    __slots__ = ("to",)
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    to: str
    def __init__(self, to: _Optional[str] = ..., **kwargs) -> None: ...
