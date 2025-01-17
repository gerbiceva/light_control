# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import grpc_server.service_pb2 as service__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class MyServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetCapabilities = channel.unary_unary(
                '/MyService/GetCapabilities',
                request_serializer=service__pb2.Void.SerializeToString,
                response_deserializer=service__pb2.Capabilities.FromString,
                _registered_method=True)
        self.GraphUpdate = channel.unary_unary(
                '/MyService/GraphUpdate',
                request_serializer=service__pb2.GraphUpdated.SerializeToString,
                response_deserializer=service__pb2.Void.FromString,
                _registered_method=True)
        self.StreamNotifications = channel.unary_stream(
                '/MyService/StreamNotifications',
                request_serializer=service__pb2.Void.SerializeToString,
                response_deserializer=service__pb2.Notification.FromString,
                _registered_method=True)


class MyServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetCapabilities(self, request, context):
        """get the list of nodes that the server supports along with their descriptions
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GraphUpdate(self, request, context):
        """Get the new edges and nodes from the frontend
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamNotifications(self, request, context):
        """Makes a phone call and communicate states via a stream.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MyServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetCapabilities': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCapabilities,
                    request_deserializer=service__pb2.Void.FromString,
                    response_serializer=service__pb2.Capabilities.SerializeToString,
            ),
            'GraphUpdate': grpc.unary_unary_rpc_method_handler(
                    servicer.GraphUpdate,
                    request_deserializer=service__pb2.GraphUpdated.FromString,
                    response_serializer=service__pb2.Void.SerializeToString,
            ),
            'StreamNotifications': grpc.unary_stream_rpc_method_handler(
                    servicer.StreamNotifications,
                    request_deserializer=service__pb2.Void.FromString,
                    response_serializer=service__pb2.Notification.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'MyService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('MyService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class MyService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetCapabilities(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MyService/GetCapabilities',
            service__pb2.Void.SerializeToString,
            service__pb2.Capabilities.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GraphUpdate(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/MyService/GraphUpdate',
            service__pb2.GraphUpdated.SerializeToString,
            service__pb2.Void.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def StreamNotifications(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/MyService/StreamNotifications',
            service__pb2.Void.SerializeToString,
            service__pb2.Notification.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
