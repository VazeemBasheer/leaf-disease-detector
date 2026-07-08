import os
import sys
import json
import time
import requests

# Set endpoint URL
BASE_URL = "http://127.0.0.1:8000"

def test_health(expected_loaded=True):
    print(f"Testing /health endpoint (expecting model_loaded={expected_loaded})...")
    url = f"{BASE_URL}/health"
    res = requests.get(url)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "ok", "Expected status ok"
    assert data["model_loaded"] is expected_loaded, f"Expected model_loaded {expected_loaded}, got {data['model_loaded']}"
    print("Health check successful!")
    print(data)
    print("-" * 50)

def test_predictions():
    print("Testing /predict image inferences...")
    samples = {
        "apple_healthy": ("samples/apple_healthy.jpg", "reports/prediction_api_apple_healthy.json"),
        "tomato_early_blight": ("samples/tomato_early_blight.jpg", "reports/prediction_api_tomato_early_blight.json"),
        "tomato_late_blight": ("samples/tomato_late_blight.jpg", "reports/prediction_api_tomato_late_blight.json")
    }
    
    for key, (path, out_path) in samples.items():
        if not os.path.exists(path):
            print(f"Skipping {key} because {path} is missing.")
            continue
            
        print(f"Submitting {path}...")
        url = f"{BASE_URL}/predict"
        
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "image/jpeg")}
            res = requests.post(url, files=files)
            
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        
        # Verify latency header
        latency = res.headers.get("X-Inference-Ms")
        assert latency is not None, "Inference latency header not returned!"
        print(f"X-Inference-Ms latency: {latency} ms")
        
        # Verify Pydantic response schema keys
        data = res.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "is_diseased" in data
        assert "probabilities" in data
        
        # Save response payload to file in UTF-8
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=4)
            
        print(f"Output saved to {out_path}")
        print(json.dumps(data, indent=2))
        print("-" * 50)

def test_invalid_mime():
    print("Testing /predict with invalid MIME type...")
    url = f"{BASE_URL}/predict"
    temp_txt_path = "tests/temp_test.txt"
    with open(temp_txt_path, "w") as f:
        f.write("Some text dataset leaf blight content")
        
    with open(temp_txt_path, "rb") as f:
        files = {"file": ("temp_test.txt", f, "text/plain")}
        res = requests.post(url, files=files)
        
    # Clean up
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
        
    assert res.status_code == 415, f"Expected 415, got {res.status_code}"
    data = res.json()
    assert "detail" in data, "Expected 'detail' key in structured error"
    assert "error_code" in data, "Expected 'error_code' key in structured error"
    assert data["error_code"] == "UNSUPPORTED_MEDIA_TYPE", f"Expected UNSUPPORTED_MEDIA_TYPE, got {data['error_code']}"
    print("Invalid mime type correctly rejected with status 415 and structured JSON!")
    print(data)
    print("-" * 50)

def test_corrupt_file():
    print("Testing /predict with corrupt image file...")
    url = f"{BASE_URL}/predict"
    temp_corrupt_path = "tests/temp_corrupt.jpg"
    with open(temp_corrupt_path, "w") as f:
        f.write("corrupt_image_data")
        
    with open(temp_corrupt_path, "rb") as f:
        files = {"file": ("temp_corrupt.jpg", f, "image/jpeg")}
        res = requests.post(url, files=files)
        
    # Clean up
    if os.path.exists(temp_corrupt_path):
        os.remove(temp_corrupt_path)
        
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    data = res.json()
    assert "detail" in data, "Expected 'detail' key in structured error"
    assert "error_code" in data, "Expected 'error_code' key in structured error"
    assert data["error_code"] == "INVALID_IMAGE_FILE", f"Expected INVALID_IMAGE_FILE, got {data['error_code']}"
    print("Corrupt file correctly rejected with status 400 and structured JSON!")
    print(data)
    print("-" * 50)

def test_offline_503():
    print("Testing 503 SERVICE UNAVAILABLE when model weights are missing...")
    env_path = ".env"
    env_backup = ".env.backup"
    
    # 1. Back up current env file
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_content = f.read()
        with open(env_backup, "w") as f:
            f.write(env_content)
    else:
        env_content = ""

    try:
        # 2. Write offline configuration environment variables
        print("Switching .env setting MODEL_PATH to non_existent.pth to trigger startup failure...")
        with open(env_path, "w") as f:
            f.write("MODEL_PATH=models/checkpoints/non_existent_weights_file.pth\n")
            f.write("DISEASE_THRESHOLD=0.35\n")
            
        # Touch app/main.py to trigger reload
        main_py_path = "app/main.py"
        with open(main_py_path, "r") as f:
            main_content = f.read()
        with open(main_py_path, "w") as f:
            f.write(main_content + "\n# Touch to reload offline\n")

        print("Waiting 3 seconds for Uvicorn to detect changes and reload...")
        time.sleep(3)

        # 3. Request health check (should return model_loaded = False)
        test_health(expected_loaded=False)

        # 4. Attempt prediction check (should return 503 with structured JSON error)
        print("Calling /predict when model is offline...")
        url = f"{BASE_URL}/predict"
        dummy_healthy_path = "samples/apple_healthy.jpg"
        if os.path.exists(dummy_healthy_path):
            with open(dummy_healthy_path, "rb") as f:
                files = {"file": ("apple_healthy.jpg", f, "image/jpeg")}
                res = requests.post(url, files=files)
            
            assert res.status_code == 503, f"Expected 503, got {res.status_code}"
            data = res.json()
            assert "detail" in data
            assert "error_code" in data
            assert data["error_code"] == "MODEL_OFFLINE", f"Expected MODEL_OFFLINE, got {data['error_code']}"
            print("Predictions correctly rejected with 503 Status Code and structured JSON:")
            print(data)
            print("-" * 50)
        else:
            print("Skipping offline prediction query because apple_healthy.jpg is missing.")
            
    finally:
        # 5. Restore backup environment
        print("Restoring original .env configuration...")
        if os.path.exists(env_backup):
            with open(env_backup, "r") as f:
                restored_content = f.read()
            with open(env_path, "w") as f:
                f.write(restored_content)
            os.remove(env_backup)
        else:
            # Fallback default configuration
            with open(env_path, "w") as f:
                f.write("MODEL_PATH=models/checkpoints/leafcnn_20260628_acc_0.992.pth\n")
                f.write("DISEASE_THRESHOLD=0.35\n")
                
        # Touch app/main.py back to recover
        with open(main_py_path, "r") as f:
            main_restored = f.read()
        # strip the touch suffix
        if "# Touch to reload offline" in main_restored:
            main_restored = main_restored.replace("\n# Touch to reload offline\n", "")
        with open(main_py_path, "w") as f:
            f.write(main_restored)

        print("Waiting 3 seconds for Uvicorn to restore state and reload...")
        time.sleep(3)
        # Check health status is recovered
        test_health(expected_loaded=True)

def main():
    try:
        test_health(expected_loaded=True)
        test_predictions()
        test_invalid_mime()
        test_corrupt_file()
        test_offline_503()
        print("ALL API VALIDATIONS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"API VALIDATION FAILED! AssertionError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"API VALIDATION FAILED! Exception: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
