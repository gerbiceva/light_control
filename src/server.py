import asyncio
import uvicorn
import grpc_server.service_pb2_grpc
import grpc_server.service_pb2
from sonora.asgi import grpcASGI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from node_loading import load_nodes

# class for handling actual communication with the client
class MyService(grpc_server.service_pb2_grpc.MyServiceServicer):
    async def GetCapabilities(self, request, context):
        nodes = load_nodes()
        nodes_message = []
        for node in nodes:
            inputs = []
            print(node[1]["Summary"][0])
            for param in node[1]["Parameters"]:
                if param.type != "None":
                    inputs.append(grpc_server.service_pb2.Port(name=param.name, type=getattr(grpc_server.service_pb2, param.type)))
            outputs = []
            for ret in node[1]["Returns"]:
                if ret.type != "None":
                    outputs.append(grpc_server.service_pb2.Port(name=ret.name, type=getattr(grpc_server.service_pb2, ret.type)))
            nodes_message.append(grpc_server.service_pb2.NodeCapability(name=node[1]["Summary"][0], description=node[1]["Extended Summary"][0], inputs=inputs, outputs=outputs))
        print(nodes_message)
        return grpc_server.service_pb2.Capabilities(nodes=nodes_message)

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
    grpc_server.service_pb2_grpc.add_MyServiceServicer_to_server(MyService(), application)

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

def start_grpc():
    asyncio.run(main())

if __name__=="__main__":
    start_grpc()
