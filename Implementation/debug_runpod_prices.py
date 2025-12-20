
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"

def run_query(query):
    payload = {'query': query}
    response = requests.post(GRAPHQL_URL, json=payload)
    if response.status_code != 200:
        return {"status": response.status_code, "text": response.text}
    return response.json()

query = """
query {
  gpuTypes {
    id
    displayName
    memoryInGb
  }
}
"""

try:
    result = run_query(query)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
