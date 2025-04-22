from pycrdt_websocket import ASGIServer, WebsocketServer
from uvicorn import Config
import uvicorn


async def server(port: int):
    websocket_server = WebsocketServer()
    app = ASGIServer(websocket_server)
    config = Config(app, port=port, log_level="info", use_colors=True)
    server = uvicorn.Server(config)
    async with websocket_server:
        await server.serve()
