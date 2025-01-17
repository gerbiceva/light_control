from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NotifType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Log: _ClassVar[NotifType]
    Success: _ClassVar[NotifType]
    Error: _ClassVar[NotifType]

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
Log: NotifType
Success: NotifType
Error: NotifType
Int: BaseType
Float: BaseType
String: BaseType
Color: BaseType
Curve: BaseType
ColorArray: BaseType
Array: BaseType
Vector2D: BaseType
Vector3D: BaseType

class Notification(_message.Message):
    __slots__ = ("title", "message", "type")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    title: str
    message: str
    type: NotifType
    def __init__(self, title: _Optional[str] = ..., message: _Optional[str] = ..., type: _Optional[_Union[NotifType, str]] = ...) -> None: ...

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
    __slots__ = ("name", "description", "inputs", "outputs", "namespace")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    inputs: _containers.RepeatedCompositeFieldContainer[Port]
    outputs: _containers.RepeatedCompositeFieldContainer[Port]
    namespace: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., namespace: _Optional[str] = ...) -> None: ...

class Capabilities(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeCapability]
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeCapability, _Mapping]]] = ...) -> None: ...

class EdgeMsg(_message.Message):
    __slots__ = ("fromNode", "fromPort", "toNode", "toPort")
    FROMNODE_FIELD_NUMBER: _ClassVar[int]
    FROMPORT_FIELD_NUMBER: _ClassVar[int]
    TONODE_FIELD_NUMBER: _ClassVar[int]
    TOPORT_FIELD_NUMBER: _ClassVar[int]
    fromNode: str
    fromPort: str
    toNode: str
    toPort: str
    def __init__(self, fromNode: _Optional[str] = ..., fromPort: _Optional[str] = ..., toNode: _Optional[str] = ..., toPort: _Optional[str] = ...) -> None: ...

class NodeMsg(_message.Message):
    __slots__ = ("id", "name", "value", "namespace")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    value: str
    namespace: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., value: _Optional[str] = ..., namespace: _Optional[str] = ...) -> None: ...

class GraphUpdated(_message.Message):
    __slots__ = ("nodes", "edges")
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeMsg]
    edges: _containers.RepeatedCompositeFieldContainer[EdgeMsg]
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeMsg, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[EdgeMsg, _Mapping]]] = ...) -> None: ...
