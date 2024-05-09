import requests
from dotenv import load_dotenv
import os
load_dotenv()


def make_prediction(format: str, topic: str) -> dict:
    # Define the payload
    payload = {
        "format": format,
        "topic": topic
    }
    
    # Make the POST request
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        response.raise_for_status()  # Raise an exception for 4XX or 5XX status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Example usage
format = "LinkedIn Post"
topic = "What is football?"
result = make_prediction(format, topic)
print(result)
