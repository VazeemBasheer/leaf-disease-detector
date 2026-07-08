# test_single.py
import sys
import requests
import json

if len(sys.argv) < 2:
    print("Usage: python test_single.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
url = "http://127.0.0.1:8000/predict"

try:
    with open(image_path, "rb") as f:
        # FastAPI's UploadFile gets parameter via 'file' key
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Latency: {response.headers.get('X-Inference-Ms')} ms")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
