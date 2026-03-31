from ml_models.svd_waymo.feature_extractor import extract_features

def detect_anomaly(data):
    features = extract_features(data["point_cloud"])
    
    score = max(features["singular_values"])
    
    return {
        "anomaly": score > 10,
        "score": score,
        "features": features
    }