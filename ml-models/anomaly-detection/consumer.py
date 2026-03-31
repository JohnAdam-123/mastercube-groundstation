from telemetry.stream import ml_queue
from ml_models.anomaly_detection.inference import detect_anomaly

def start_ml_consumer():
    while True:
        data = ml_queue.get()

        result = detect_anomaly(data)

        if result["anomaly"]:
            print("⚠️ Anomaly detected:", result)