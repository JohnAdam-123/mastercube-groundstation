import asyncio
import websockets

async def test():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to server")

        while True:
            data = await websocket.recv()
            print("📡 Received:", data)

asyncio.run(test())