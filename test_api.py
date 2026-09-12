import base64
import requests
import glob

# Grab the first wav file in the audio directory
test_file = glob.glob("audio/*.wav")[0]

with open(test_file, "rb") as f:
    encoded_audio = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://127.0.0.1:8000/detect", 
    json={"audio_base64": encoded_audio}
)

print(f"Testing {test_file}")
print("Status:", response.status_code)
print("Result:", response.json())