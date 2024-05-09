import requests

def make_prediction(url: str, question: str) -> dict:
    # Define the payload
    payload = {
        "url": url,
        "question": question
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
url = "https://en.wikipedia.org/wiki/Generative_artificial_intelligence"
question = "What is football?"
result = make_prediction(url, question)
print(result)
