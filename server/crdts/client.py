from httpx_ws import aconnect_ws
from pycrdt import Doc, Array, Map, ArrayEvent
from pycrdt_websocket import WebsocketProvider
import asyncio
from pycrdt_websocket.websocket import HttpxWebsocket

ydoc = Doc()  # dont ever at all under any circumstance observe this fucker.
yarray = ydoc.get("count", type=Array)


def cb(event: ArrayEvent):
    try:
        print(event)
        pass

    except Exception as e:
        print("except", e)


async def client(port: int, room_name: str):
    async with (
        aconnect_ws(f"ws://localhost:{port}/{room_name}") as websocket,
        WebsocketProvider(ydoc, HttpxWebsocket(websocket, room_name)),
    ):
        # Changes to remote ydoc are applied to local ydoc.
        # Changes to local ydoc are sent over the WebSocket and
        # broadcast to all clients.
        yarray.observe(cb)
        await asyncio.Future()
