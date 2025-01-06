import asyncio
import uvicorn
import service_pb2_grpc
import service_pb2
from sonora.asgi import grpcASGI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# class for handling actual communication with the client
class MyService(service_pb2_grpc.MyServiceServicer):

    def GetCapabilities(self, request, context):
        """get the list of nodes that the server supports along with their descriptions
        """
        raise NotImplementedError('Method not implemented!')

    def NodesUpdated(self, request, context):
        """Get the new edges and nodes from the frontend
        """
        raise NotImplementedError('Method not implemented!')


class GRPCWebMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Type"] = "application/grpc-web+proto"

        # modifying raw headers is necesarry
        del response.raw_headers[0]
        return response

async def main():
    application = grpcASGI(uvicorn, False)
    # Attach your gRPC server implementation.
    service_pb2_grpc.add_MyServiceServicer_to_server(MyService(), application)

     # Add CORS middleware
    application = CORSMiddleware(
        application,
        allow_origins=["*"],  # Adjust this to your allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wrap the app with GRPCWebMiddleware
    application = GRPCWebMiddleware(application)

    config = uvicorn.Config(application, port=50051, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())