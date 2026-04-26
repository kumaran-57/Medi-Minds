# gan/generate_images.py

import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow to prevent import errors with transformers

import torch
import torch.nn as nn
from torchvision.utils import save_image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# === Configs ===
latent_dim = 100
output_dir = "static/generated_data/"
checkpoint_path = "static/gan_generator.pth"

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# === Load Generator ===
netG = Generator().to(device)
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# === Generate Images ===
def generate_synthetic_images(num_images):
    saved_paths=[]
  # Number of synthetic images to generate
    for i in range(num_images):
        z = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_img = netG(z)
        save_path = os.path.join(output_dir, f"synthetic_{i+1:04d}.png")
        saved_paths.append(save_path.replace("static",""))
        save_image(fake_img, save_path, normalize=True)
    print(f"Successfully generated {num_images} synthetic images at: {output_dir}")
    print(saved_paths)
    return saved_paths

model_name = "dima806/chest_xray_pneumonia_detection"
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        print("Loading ViT model from Hugging Face...")
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        print("ViT model loaded successfully!")

def model_predict(path):
    load_model()
    image = Image.open(path).convert("RGB")

        # Apply the processor (resizing, rescaling, normalization, etc.)
    inputs = processor(images=image, return_tensors="pt")

        # Run model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Get class label
    predicted_label  = model.config.id2label[predicted_class_idx]
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item() * 100

    return predicted_label,confidence