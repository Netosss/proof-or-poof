import os
import json
from google import genai
from google.genai import types
from PIL import Image

def extract_filenames(image_path):
    print(f"Reading image from: {image_path}")
    
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=60000)
    )

    try:
        img = Image.open(image_path)
        
        prompt = """
        List all the filenames visible in this image. 
        Return ONLY a JSON array of strings. 
        Example: ["file1.jpg", "image2.png"]
        If you see file extensions, include them. 
        If not, just the names.
        """
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        print("Raw response:", response.text)
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    screenshot_path = "/Users/netanel.ossi/.cursor/projects/Users-netanel-ossi-Desktop-proof-or-poof/assets/Screenshot_2026-02-22_at_9.35.07-1c854973-628c-40b9-b946-204db9e63987.png"
    filenames = extract_filenames(screenshot_path)
    print("\nExtracted Filenames:")
    print(json.dumps(filenames, indent=2))
