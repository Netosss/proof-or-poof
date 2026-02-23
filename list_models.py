import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # The new SDK doesn't have list_models directly on client sometimes?
    # Let's try to find how to list models or just try a simple generation
    print("Listing models...")
    # The SDK structure is client.models.list()
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
