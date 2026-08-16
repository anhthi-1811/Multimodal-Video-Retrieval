"""
=============================================================================
EMBEDDING MODELS (FEATURE ENCODING)
=============================================================================
Description:
Contains encoder classes that transform raw data (Text, Images) into 
high-dimensional numerical vectors. These classes operate independently 
and do not interact with any databases.
=============================================================================
"""

import os
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import warnings

# Suppress unnecessary warning logs for a cleaner terminal
warnings.filterwarnings("ignore")

class TextEncoder:
    """
    Dedicated encoder for general text processing (OCR, BLIP Captions, ASR).
    Utilizes BGE-M3, which is highly optimized for multilingual text (including Vietnamese).
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        # Automatically select GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TextEncoder ({model_name}) on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str) -> list:
        """
        Transforms a string into a normalized dense vector.
        """
        if not text or not text.strip():
            return []
        
        # Encode text and normalize the embeddings for Cosine Similarity
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class ObjectEncoder:
    """
    Dedicated encoder for YOLO object labels.
    Utilizes the Text Model of CLIP to ensure the output vector shares the 
    same semantic space as the physical images.
    """
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ObjectEncoder ({model_name}) on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval() # Set model to evaluation mode

    def encode(self, text: str) -> list:
        """
        Transforms object label strings (e.g., '2 person, 1 racket') into a vector.
        """
        if not text or not text.strip():
            return []

        try:
            # Tokenize the input text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                # Extract text features using CLIP
                outputs = self.model.get_text_features(**inputs)
            
            # Normalize the vector (L2 Normalization)
            vector = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return vector.squeeze().tolist()
            
        except Exception as e:
            print(f"Error encoding object text '{text}': {e}")
            return []


class ImageEncoder:
    """
    Dedicated encoder for physical keyframe images.
    Utilizes the Vision Model of CLIP.
    """
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ImageEncoder ({model_name}) on {self.device}...")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode(self, image_path: str) -> list:
        """
        Reads an image file from the disk and transforms it into a vector.
        """
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return []

        try:
            # Load and convert image to standard RGB format
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Extract image features using CLIP
                outputs = self.model.get_image_features(**inputs)
            
            # Normalize the vector
            vector = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return vector.squeeze().tolist()
            
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return []  