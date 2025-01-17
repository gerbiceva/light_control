# GRPC
import asyncio
import uvicorn
import grpc_server.service_pb2_grpc
import grpc_server.service_pb2
from sonora.asgi import grpcASGI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# WEB
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# OTHER SHIT
import inspect
import importlib
import os
from numpydoc.docscrape import FunctionDoc
from graph import Graph
import re
import colorsys
import jax.numpy as jnp
from utils import FrameLimiter
from typing import Callable
from collections import defaultdict


defined_nodes: dict[str:Callable] = {}
supplementary: list[Callable] = []
graph = Graph()


# class for handling actual communication with the client
class MyService(grpc_server.service_pb2_grpc.MyServiceServicer):
    async def GetCapabilities(self, request, context):
        global defined_nodes
        # Path to the directory containing the files
        directory = "nodes"

        # List to store the imported modules
        modules = []
        namespaces = []

        # Iterate over all .py files in the directory
        for filename in os.listdir(directory):
            if (
                filename.endswith(".py")
                and filename != "__init__.py"
                and not filename.startswith("_")
            ):
                module_name = filename[:-3]  # Remove the .py extension
                module = importlib.import_module(f"{directory}.{module_name}")
                modules.append(module)
                namespaces.append(module_name)

        nodes = []
        for namespace, module in zip(namespaces, modules):
            for member in inspect.getmembers(module):
                if (
                    member[0][0] != "_"
                    and inspect.isfunction(member[1])
                    and hasattr(member[1], "__is_node__")
                ):
                    nodes.append((member[1], FunctionDoc(member[1]), namespace))
                if hasattr(member[1], "__each_tick__"):
                    supplementary.append(member[1])
        nodes_message = []
        for node in nodes:
            inputs = []
            # print(node[1]["Summary"][0])
            defined_nodes[f"{node[2]}/{node[1]['Summary'][0]}"] = node[0]
            if hasattr(node[0], "__primitive__"):
                # print(node)
                continue
            for param in node[1]["Parameters"]:
                if param.type != "None":
                    inputs.append(
                        grpc_server.service_pb2.Port(
                            name=param.name,
                            type=getattr(grpc_server.service_pb2, param.type),
                        )
                    )
            outputs = []
            for ret in node[1]["Returns"]:
                if ret.type != "None":
                    outputs.append(
                        grpc_server.service_pb2.Port(
                            name=ret.name,
                            type=getattr(grpc_server.service_pb2, ret.type),
                        )
                    )
            nodes_message.append(
                grpc_server.service_pb2.NodeCapability(
                    name=node[1]["Summary"][0],
                    description=node[1]["Extended Summary"][0],
                    inputs=inputs,
                    outputs=outputs,
                    namespace=node[2],
                )
            )
        # print(nodes_message)
        return grpc_server.service_pb2.Capabilities(nodes=nodes_message)

    async def GraphUpdate(self, request, context):
        # print(request.nodes)
        # print(request.edges)
        nodes = {}
        for requested in request.nodes:
            f = defined_nodes[f"{requested.namespace}/{requested.name}"]
            if hasattr(f, "__initialize__"):
                made_f = f()
                made_f.__doc__ = f.__doc__
                nodes[requested.id] = made_f
            elif hasattr(f, "__primitive__"):
                if requested.name == "Int":
                    try:
                        nodes[requested.id] = f(int(requested.value))
                    except ValueError:
                        nodes[requested.id] = f(0)
                elif requested.name == "Float":
                    try:
                        nodes[requested.id] = f(float(requested.value))
                    except ValueError:
                        nodes[requested.id] = f(0.0)
                elif requested.name == "String":
                    try:
                        nodes[requested.id] = f(str(requested.value))
                    except ValueError:
                        nodes[requested.id] = f("")
                elif requested.name == "Color":
                    try:
                        if requested.value == "":
                            raise ValueError
                        hsl = list(map(int, re.findall(r"\d+", requested.value)))

                        hsl[0] = hsl[0] / 360.0
                        hsl[1] = hsl[1] / 100.0
                        hsl[2] = hsl[2] / 100.0

                        nodes[requested.id] = f(
                            jnp.array(
                                colorsys.rgb_to_hsv(
                                    *colorsys.hls_to_rgb(hsl[0], hsl[2], hsl[1])
                                )
                            )
                        )
                    except ValueError:
                        nodes[requested.id] = f(jnp.array([0.0, 0.0, 0.0]))
                elif requested.name == "Curve":
                    try:
                        print(requested.value)
                        nodes[requested.id] = f(str(requested.value))
                    except ValueError:
                        nodes[requested.id] = f("")
            else:
                nodes[requested.id] = f
        edges = defaultdict(list)
        for edge in request.edges:
            edges[(edge.fromNode, edge.fromPort)] += [(edge.toNode, edge.toPort)]
        try:
            graph.construct(nodes, edges)
        except Exception as e:
            print(e)
        # print(graph.edges)
        return grpc_server.service_pb2.Void()


class GRPCWebMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Type"] = "application/grpc-web+proto"

        # modifying raw headers is necesarry
        del response.raw_headers[0]
        return response


async def webUI(port: int):
    webApp = FastAPI()
    webApp.mount("/", StaticFiles(directory="dist", html=True), name="ui")

    config = uvicorn.Config(webApp, port=port, log_level="info", log_config={})
    server = uvicorn.Server(config)
    await server.serve()


async def grpc(port: int):
    application = grpcASGI(uvicorn, False)
    # Attach your gRPC server implementation.
    grpc_server.service_pb2_grpc.add_MyServiceServicer_to_server(
        MyService(), application
    )

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

    config = uvicorn.Config(application, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def loop():
    limiter = FrameLimiter(30)
    while True:
        try:
            graph.evaluate()
        except Exception as e:
            print(e)
            continue

        for f in supplementary:
            f()
        await limiter.tick()


async def start_server():
    loop_task = asyncio.create_task(loop())
    grpc_task = asyncio.create_task(grpc(50051))
    web_task = asyncio.create_task(webUI(8080))
    await asyncio.gather(grpc_task, loop_task, web_task)
    # await grpc_task


if __name__ == "__main__":
    asyncio.run(start_server())
