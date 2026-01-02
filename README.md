# AI Provenance Detection API

Deterministic AI detection for images and videos using C2PA metadata.

## Goal
Build a production-ready Python server that:
1. Takes an image or video as input.
2. Determines deterministically if it was AI-generated or AI-edited using C2PA metadata.
3. Returns the generator/provider (OpenAI, Google, Adobe, etc.) if available.
4. Handles missing metadata gracefully (returns `is_ai: null`).

## Project Structure
- `app/`: Core API and detection logic.
- `third_party/`: Contains `c2pa-python` as a Git submodule.
- `requirements.txt`: Python dependencies.

## Setup
1. Clone the repository with submodules:
   ```bash
   git clone --recursive <repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python app/main.py
   ```

## API Usage
`POST /detect`
Accepts a multipart form-data file (image or video).

Returns:
```json
{
  "is_ai": true,
  "provider": "OpenAI DALL-E 3",
  "method": "c2pa",
  "confidence": 1.0
}
```
If no metadata is found:
```json
{
  "is_ai": null,
  "provider": null,
  "method": "none",
  "confidence": 0.0
}
```



