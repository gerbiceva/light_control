from httpx_ws import aconnect_ws
from pycrdt import Doc, Array, Map, MapEvent
from pycrdt_websocket import WebsocketProvider
import asyncio
from pycrdt_websocket.websocket import HttpxWebsocket


ydoc = Doc()
syncedAppState = ydoc.get("syncedAppState", type=Map)


def cb(events):
    for event in events:
        event: MapEvent = event
        # print(f"Target: {event.target}")
        # print(f"kys:{event.keys}")
        print("change")


async def client(port: int, room_name: str):
    async with (
        aconnect_ws(f"ws://localhost:{port}/{room_name}") as websocket,
        WebsocketProvider(ydoc, HttpxWebsocket(websocket, room_name)) as provider,
    ):
        print(
            "Connection state:",
            websocket.connection.state,
            "started" if provider._started else "not started",
        )

        # Initialize the state using dict notation
        with ydoc.transaction():
            syncedAppState["state"] = map0 = Map(
                {
                    "subgraphs": Map(),
                    "main": Map(
                        {
                            "id": 0,
                            "name": "main",
                            "description": "This main flow gets executed always",
                            "nodes": Array(),
                            "edges": Array(),
                        }
                    ),
                }
            )

        print("Initialized state with CRDT types:")
        print(map0.to_py())

        # Observe changes
        syncedAppState.observe_deep(cb)

        # Keep the connection open
        await asyncio.Future()
