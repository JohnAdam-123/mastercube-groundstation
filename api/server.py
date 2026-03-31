import asyncio
from fastapi import FastAPI, WebSocket

app = FastAPI()

# Store connected clients
clients = []

# Store incoming data
latest_data = None


# 🔹 INPUT WebSocket (simulate_data connects here)
@app.websocket("/ws/input")
async def input_ws(websocket: WebSocket):
    await websocket.accept()
    print("🟡 Simulator connected")

    global latest_data

    try:
        while True:
            data = await websocket.receive_json()
            latest_data = data
            print("📥 Received from simulator:", data)

    except Exception as e:
        print("Input error:", e)


# 🔹 OUTPUT WebSocket (test_ws connects here)
@app.websocket("/ws")
async def output_ws(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Client connected")

    try:
        while True:
            if latest_data:
                await websocket.send_json(latest_data)
                print("📡 Sent to client:", latest_data)  # 👈 ADD THIS

            await asyncio.sleep(1)

    except Exception as e:
        print("Output error:", e)
        clients.remove(websocket)