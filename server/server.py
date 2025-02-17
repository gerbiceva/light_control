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
from graph import Graph
import re
import colorsys
import jax.numpy as jnp
from utils import FrameLimiter
from typing import Callable
from collections import defaultdict
from node_loading import load_nodes
from numpydoc.docscrape import FunctionDoc

defined_nodes: dict[str:Callable] = {}
each_tick: list[Callable] = []
threads: list[Callable] = []
running_threads: list[Callable] = []
graph = Graph({}, {})

# class for handling actual communication with the client
class MyService(grpc_server.service_pb2_grpc.MyServiceServicer):
    async def GetCapabilities(self, request, context):
        # nodes, threads, each_tick = load_nodes('nodes')

        nodes_message = []
        for key, node in defined_nodes.items():
            node_definition = FunctionDoc(node)
            namespace = key.split(sep='/')[0]
            inputs = []
            if hasattr(node, "__primitive__"):
                continue
            for param in node_definition["Parameters"]:
                if param.type != "None":
                    inputs.append(
                        grpc_server.service_pb2.Port(
                            name=param.name,
                            type=getattr(grpc_server.service_pb2, param.type),
                        )
                    )
            outputs = []
            for ret in node_definition["Returns"]:
                if ret.type != "None":
                    outputs.append(
                        grpc_server.service_pb2.Port(
                            name=ret.name,
                            type=getattr(grpc_server.service_pb2, ret.type),
                        )
                    )
            nodes_message.append(
                grpc_server.service_pb2.NodeCapability(
                    name=node_definition["Summary"][0],
                    description=node_definition["Extended Summary"][0],
                    inputs=inputs,
                    outputs=outputs,
                    namespace=namespace,
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
            for thread in running_threads:
                thread.stop()
            global graph
            graph = Graph(nodes, edges)
            for f in threads:
                running_threads.append(f())
        except Exception as e:
            print(e)
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

    config = uvicorn.Config(webApp, port=port, log_level="info")
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


def loop():
    limiter = FrameLimiter(60)
    while True:
        try:
            graph.evaluate()
        except Exception as e:
            print(e)
            continue

        for f in each_tick:
            f()
        limiter.tick()

def generating(generator):
    global defined_nodes
    for node in generator():
        defined_nodes = defined_nodes | {f"dynamic/{FunctionDoc(node)['Summary'][0]}": node}

async def start_server():
    global defined_nodes, threads, each_tick
    defined_nodes, threads, each_tick, generators = load_nodes('nodes')
    generator_threads = [asyncio.to_thread(generating, generator) for generator in generators]

    loop_thread = asyncio.to_thread(loop)
    grpc_task = asyncio.create_task(grpc(50051))
    web_task = asyncio.create_task(webUI(8080))
    await asyncio.gather(grpc_task, loop_thread, web_task, *generator_threads)
    # await grpc_task


if __name__ == "__main__":
    asyncio.run(start_server())
