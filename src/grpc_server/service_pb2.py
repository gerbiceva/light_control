# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: service.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rservice.proto\"\x06\n\x04Void\"-\n\x04Port\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x17\n\x04type\x18\x03 \x01(\x0e\x32\t.BaseType\"u\n\x0eNodeCapability\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x15\n\x06inputs\x18\x04 \x03(\x0b\x32\x05.Port\x12\x16\n\x07outputs\x18\x05 \x03(\x0b\x32\x05.Port\x12\x11\n\tnamespace\x18\x06 \x01(\t\".\n\x0c\x43\x61pabilities\x12\x1e\n\x05nodes\x18\x01 \x03(\x0b\x32\x0f.NodeCapability\"M\n\x07\x45\x64geMsg\x12\x10\n\x08\x66romNode\x18\x01 \x01(\t\x12\x10\n\x08\x66romPort\x18\x02 \x01(\t\x12\x0e\n\x06toNode\x18\x03 \x01(\t\x12\x0e\n\x06toPort\x18\x04 \x01(\t\"T\n\x07NodeMsg\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x12\n\x05value\x18\x03 \x01(\tH\x00\x88\x01\x01\x12\x11\n\tnamespace\x18\x04 \x01(\tB\x08\n\x06_value\"@\n\x0cGraphUpdated\x12\x17\n\x05nodes\x18\x01 \x03(\x0b\x32\x08.NodeMsg\x12\x17\n\x05\x65\x64ges\x18\x02 \x03(\x0b\x32\x08.EdgeMsg*w\n\x08\x42\x61seType\x12\x07\n\x03Int\x10\x00\x12\t\n\x05\x46loat\x10\x02\x12\n\n\x06String\x10\x03\x12\t\n\x05\x43olor\x10\x04\x12\t\n\x05\x43urve\x10\x05\x12\x0e\n\nColorArray\x10\x06\x12\t\n\x05\x41rray\x10\x07\x12\x0c\n\x08Vector2D\x10\x08\x12\x0c\n\x08Vector3D\x10\t2Y\n\tMyService\x12\'\n\x0fGetCapabilities\x12\x05.Void\x1a\r.Capabilities\x12#\n\x0bGraphUpdate\x12\r.GraphUpdated\x1a\x05.Voidb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_BASETYPE']._serialized_start=470
  _globals['_BASETYPE']._serialized_end=589
  _globals['_VOID']._serialized_start=17
  _globals['_VOID']._serialized_end=23
  _globals['_PORT']._serialized_start=25
  _globals['_PORT']._serialized_end=70
  _globals['_NODECAPABILITY']._serialized_start=72
  _globals['_NODECAPABILITY']._serialized_end=189
  _globals['_CAPABILITIES']._serialized_start=191
  _globals['_CAPABILITIES']._serialized_end=237
  _globals['_EDGEMSG']._serialized_start=239
  _globals['_EDGEMSG']._serialized_end=316
  _globals['_NODEMSG']._serialized_start=318
  _globals['_NODEMSG']._serialized_end=402
  _globals['_GRAPHUPDATED']._serialized_start=404
  _globals['_GRAPHUPDATED']._serialized_end=468
  _globals['_MYSERVICE']._serialized_start=591
  _globals['_MYSERVICE']._serialized_end=680
# @@protoc_insertion_point(module_scope)
