import requests
import json

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3.5:latest",
    "messages": [
        {
            "role": "user",
            "content": "What is RAG? Answer in two short sentences."
        }
    ],
    "stream": False,
    "think": False,
    "options": {
        "num_predict": 256,
        "temperature": 0
    }
}

response = requests.post(url, json=payload, timeout=300)
response.raise_for_status()

data = response.json()

print("Message content:", repr(data.get("message", {}).get("content")))
print("Thinking:", repr(data.get("message", {}).get("thinking")))
print("Done reason:", data.get("done_reason"))
print("Full response:")
print(json.dumps(data, indent=2))