from httpx_ws import aconnect_ws
from pycrdt import Doc, Array, Map, TransactionEvent
from pycrdt_websocket import ASGIServer, WebsocketProvider, WebsocketServer
from uvicorn import Config
import uvicorn
import asyncio
from pycrdt_websocket.websocket import HttpxWebsocket


async def crdt_server(port: int):
    websocket_server = WebsocketServer()
    app = ASGIServer(websocket_server)
    config = Config(app, port=port, log_level="info", use_colors=True)
    server = uvicorn.Server(config)
    # await server.serve()
    async with websocket_server:
        await server.serve()


ydoc = Doc()  # dont ever at all under any circumstance observe this fucker.
yarray = ydoc.get("count", type=Array)


def cb(event: TransactionEvent):
    try:
        print(yarray.to_py())
        pass

    except Exception as e:
        print("except", e)


async def client(port: int, room_name: str):
    async with (
        aconnect_ws(f"ws://localhost:{port}/{room_name}") as websocket,
        WebsocketProvider(ydoc, HttpxWebsocket(websocket, room_name)) as provider,
    ):
        print(
            "connection state:",
            websocket.connection.state,
            "started" if provider._started else "kurac",
        )

        # Changes to remote ydoc are applied to local ydoc.
        # Changes to local ydoc are sent over the WebSocket and
        # broadcast to all clients.
        yarray.observe(cb)
        await asyncio.Future()
