# AI Detector API - Integration Guide

## Overview
This API provides high-accuracy AI image detection using a **Resolution-Aware Router**. It automatically distinguishes between low-resolution web images (memes, thumbnails) and high-resolution photography to apply the best model for the job.

**Infrastructure**: RunPod Serverless (GPU: RTX 4090 / L40S recommended).

---

## 1. Request Format
The API accepts **JSON** payloads via POST. It supports both single-image and batch processing.

**Endpoint**: `https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync` (or `/run` for async)

### Headers
```http
Content-Type: application/json
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

### Body (Batch Mode - Recommended)
Send a list of Base64-encoded strings. The system processes them in parallel.
```json
{
  "input": {
    "images": [
      "BASE64_STRING_IMAGE_1...",
      "BASE64_STRING_IMAGE_2..."
    ]
  }
}
```

### Body (Single Image)
```json
{
  "input": {
    "image": "BASE64_STRING..."
  }
}
```

---

## 2. Response Format
The API returns a `results` array matching the input order.

### Success Response
```json
{
  "delayTime": 12,
  "executionTime": 450,
  "id": "run-12345...",
  "status": "COMPLETED",
  "output": {
    "results": [
      {
        "ai_score": 0.98,              // Raw Probability (0.0 = Real, 1.0 = AI)
        "confidence": 0.98,            // Final Confidence Score
        "label": "AI",                 // Verdict: "AI" or "REAL"
        "router": "HighRes_Ensemble",  // trace: "HighRes_Ensemble" or "LowRes_ModelC"
        "cache_hit": false,            // true if served from memory cache
        "model_breakdown": {           
           "A": 0.99,                  // Model A Score (if Ensemble used)
           "B": 0.90                   // Model B Score (if Ensemble used)
        }
      },
      {
        "ai_score": 0.05,
        "confidence": 0.05,
        "label": "REAL",
        "router": "LowRes_ModelC",     // Handled by specialized ViT model
        "model_breakdown": {
           "C": 0.05
        }
      }
    ],
    "timing_ms": {
      "decode": 25.5,    // Time to decode Base64
      "total": 450.2,    // Total processing time
      "cache_hits": 0    // Number of images served from cache
    }
  }
}
```

---

## 3. Key Features for Integration

### A. Resolution Routing (Automatic)
*   **< 200k pixels** (e.g. 300x300): Routed to **Model C** (Specialized Low-Res Model).
    *   *Why?* Standard models fail on blurry/pixelated images. our Model C is 99% accurate here.
*   **>= 200k pixels**: Routed to **Ensemble A+B** (High-Res Experts).
    *   *Why?* Combinines "Model A" (Real Expert) and "Model B" (AI Expert) for maximum reliability.

### B. Smart Caching
*   The worker caches results by MD5 hash.
*   Retrying the same image is instant (0ms GPU time).
*   **Tip**: You don't need to implement client-side caching for short-term retries.

### C. Error Handling
If an image fails (e.g., corrupt bytes), the result for that index will conform to:
```json
{
  "error": "Image decode error: ...",
  "ai_score": 0.5
}
```
**Action**: Check `results[i].error` before reading scores.

---

## 4. Best Practices
1.  **Use Batches**: Sending 10 images in one request is significantly faster than 10 separate requests due to parallel overhead reduction.
2.  **Compress First (Optional)**: If bandwidth is an issue, resizing huge >4k images to ~2000px before sending saves network time (the model resizes internally anyway).
3.  **Keep it Warm**: RunPod cold starts take ~10s. Keep at least 1 active worker if latency is critical.
