from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch

import os

def download_models():
    model_name = "dima806/chest_xray_pneumonia_detection"
    cache_dir = "./model_cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Pre-downloading {model_name} into local folder: {cache_dir}...")
    
    # Download processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    processor.save_pretrained(cache_dir)
    
    # Download model
    model = AutoModelForImageClassification.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        dtype=torch.bfloat16
    )
    model.save_pretrained(cache_dir)
    print("Download successfully cached locally!")

if __name__ == "__main__":
    download_models()
