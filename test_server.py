import requests
import time

# Wait a moment for the server to start
time.sleep(5)

# Test if server is running
try:
    response = requests.get("http://localhost:8080")
    print(f"Server is running. Status code: {response.status_code}")
except requests.exceptions.ConnectionError as e:
    print(f"Server is not accessible: {e}")
except Exception as e:
    print(f"An error occurred: {e}")