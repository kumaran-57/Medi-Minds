from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch

def download_models():
    model_name = "dima806/chest_xray_pneumonia_detection"
    print(f"Pre-downloading {model_name} from Hugging Face...")
    
    # Download processor
    ViTImageProcessor.from_pretrained(model_name)
    
    # Download model
    AutoModelForImageClassification.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        dtype=torch.bfloat16
    )
    print("Download successfully cached!")

if __name__ == "__main__":
    download_models()
