"""
=============================================================================
EMBEDDING MODELS (FEATURE ENCODING)
=============================================================================
Description:
Contains encoder classes that transform raw data (Text, Images) into 
high-dimensional numerical vectors.
=============================================================================
"""

import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")

class TextEncoder:
    """
    Dedicated encoder for general text processing (OCR, BLIP Captions, ASR).
    Utilizes BGE-M3, highly optimized for multilingual text.
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TextEncoder ({model_name}) on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str) -> list:
        if not text or not text.strip():
            return []
        vector = self.model.encode(text, normalize_embeddings=True) 
        return vector.tolist()


class CLIPEncoder:
    """
    Unified encoder for both Image (Keyframes) and Object Labels (YOLO).
    Loads the CLIP model ONCE to optimize VRAM usage.
    """
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIPEncoder ({model_name}) on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_text(self, text: str) -> list:
        """Transforms English text (like YOLO objects) into a vector."""
        if not text or not text.strip():
            return []
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            vector = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return vector.squeeze().tolist()
        except Exception as e:
            print(f"Error encoding object text '{text}': {e}")
            return []

    def encode_image(self, image_path: str) -> list:
        """Reads an image file and transforms it into a vector."""
        if not os.path.exists(image_path):
            return []
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            vector = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return vector.squeeze().tolist()
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return [] 