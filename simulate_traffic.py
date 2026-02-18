import requests
import time
import random
import sys
from PIL import Image
import io
import threading

API_URL = "http://localhost:8000"

def generate_random_image():
    # Generate a random RGB image
    img = Image.new('RGB', (224, 224), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def send_request(request_id):
    try:
        img_bytes = generate_random_image()
        files = {'file': (f'random_{request_id}.jpg', img_bytes, 'image/jpeg')}
        
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", files=files)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            print(f"Request {request_id}: Success | Latency: {latency:.4f}s | Result: {response.json()['label']}")
        else:
            print(f"Request {request_id}: Failed | Status: {response.status_code}")
            
    except Exception as e:
        print(f"Request {request_id}: Error: {e}")

def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            print("\n--- Current Metrics ---")
            print(response.json())
            print("-----------------------\n")
    except Exception as e:
        print(f"Error fetching metrics: {e}")

if __name__ == "__main__":
    print(f"Starting traffic simulation against {API_URL}...")
    
    # Check if service is up
    try:
        requests.get(f"{API_URL}/health")
    except:
        print("Service not reachable. Is it running? (uvicorn app.main:app --reload)")
        sys.exit(1)

    # Simulate 50 requests
    for i in range(50):
        send_request(i)
        time.sleep(random.uniform(0.1, 0.5))
        
        if i % 10 == 0:
            get_metrics()
            
    get_metrics()
    print("Simulation complete.")
