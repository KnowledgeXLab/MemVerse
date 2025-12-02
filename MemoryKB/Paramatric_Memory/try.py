import requests

API_URL = "http://localhost:8001/generate"
data = {
    "query": "what is Parametric Memory",
    "max_new_tokens": 200,
    "temperature": 0.6
}

response = requests.post(API_URL, json=data)
result = response.json()
print("Query:", result["query"])
print("Response:", result["response"])