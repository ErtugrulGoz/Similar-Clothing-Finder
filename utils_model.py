import torch
import requests
import io
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoProcessor
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OPTIONS = {
    "Patrick FashionCLIP": "patrickjohncyh/fashion-clip",
    "Marqo FashionCLIP": "Marqo/marqo-fashionCLIP",
    "FashionCLIP 2.0": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "OpenCLIP ViT-H-14": "open_clip/ViT-H-14-laion2B-s32B-b79K",
    "Marqo FashionSigLIP": "Marqo/marqo-fashionSigLIP",
}

def load_selected_model(model_key):
    model_name = MODEL_OPTIONS[model_key]
    if model_key == "OpenCLIP ViT-H-14":
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
        )
        return model, preprocess
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor

def load_image_from_url(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if "image" in response.headers.get("Content-Type", "").lower():
            return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        return None

def get_image_embedding(image, model, processor):
    try:
        image_tensor = processor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
    except Exception:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()
