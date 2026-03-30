import asyncio
import websockets
import json
import time
import random

async def send_data():
    uri = "ws://127.0.0.1:8000/ws/input"

    async with websockets.connect(uri) as websocket:
        print("🟡 Connected to server (input)")

        while True:
            data = {
                "timestamp": time.time(),
                "signal_strength": random.random(),
                "satellite_id": 1234
            }

            await websocket.send(json.dumps(data))
            print("📤 Sent:", data)

            await asyncio.sleep(1)

asyncio.run(send_data())