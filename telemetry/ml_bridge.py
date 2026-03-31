from ml_models.anomaly_detection.inference import detect_anomaly

def process_for_ml(data):
    result = detect_anomaly(data)
    
    if result["anomaly"]:
        print("⚠️ Anomaly detected:", result)