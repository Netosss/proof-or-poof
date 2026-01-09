
import logging
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("builder")

def download_models():
    models = [
        "haywoodsloan/ai-image-detector-dev-deploy",
        "Ateeqq/ai-vs-human-image-detector",
        "dima806/ai_vs_real_image_detection"
    ]
    
    logger.info("Starting build-time model download...")
    
    for model_id in models:
        logger.info(f"Downloading: {model_id}")
        try:
            # Download Processor
            AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            
            # Download Model (FP16/SafeTensors usually handled by auto)
            # We load with float16 to ensure we get the right shards if they exist specifically,
            # though usually all are downloaded. 
            AutoModelForImageClassification.from_pretrained(
                model_id, 
                torch_dtype=torch.float16
            )
            logger.info(f"Successfully downloaded {model_id}")
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            raise

if __name__ == "__main__":
    download_models()
