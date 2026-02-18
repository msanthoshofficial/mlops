import requests
import time
import sys

API_URL = "http://localhost:8000"

def test_health():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("Health check passed.")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction():
    # Create a dummy image for testing
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    
    try:
        response = requests.post(f"{API_URL}/predict", files=files)
        if response.status_code == 200:
            print(f"Prediction check passed: {response.json()}")
            return True
        else:
            print(f"Prediction check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction check failed: {e}")
        return False

if __name__ == "__main__":
    print("Waiting for service to start...")
    time.sleep(5) # Give it a moment if started just now
    
    if test_health() and test_prediction():
        print("Smoke tests passed!")
        sys.exit(0)
    else:
        print("Smoke tests failed!")
        sys.exit(1)
