import asyncio
import uvicorn

# WEB
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import WebSocket
from node_graph import main_loop, grpc

input_values = {}
input_lock = asyncio.Lock()
# Modify the webUI function to include WebSocket handling
async def webUI(port: int):
    webApp = FastAPI()

    @webApp.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                if len(data) != 2:
                    continue

                input_index = data[0]
                value = max(0, min(data[1], 100))

                async with input_lock:  # Using async lock
                    input_values[input_index] = value
                    # print(f"Received input {input_index}: {value}")

        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await websocket.close()
            print("WebSocket connection closed")

    webApp.mount("/", StaticFiles(directory="dist", html=True), name="ui")

    config = uvicorn.Config(webApp, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def start_server():
    loop_thread = asyncio.to_thread(main_loop)
    grpc_task = asyncio.create_task(grpc(50051))
    web_task = asyncio.create_task(webUI(8080))
    await asyncio.gather(grpc_task, loop_thread, web_task)
    # await grpc_task


if __name__ == "__main__":
    asyncio.run(start_server())
