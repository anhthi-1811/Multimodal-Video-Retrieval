"""
=============================================================================
BLIP PROCESSOR (FEATURE EXTRACTION)
=============================================================================
Description:
Contains the BlipProcessor class. Responsible ONLY for initializing the 
BLIP image captioning model and generating a text description for a single image.
Does not handle database connections, multiprocessing, or file system loops.
=============================================================================
"""

import os
import warnings
from PIL import Image
from transformers import BlipProcessor as HuggingFaceBlipProcessor
from transformers import BlipForConditionalGeneration

# Suppress unnecessary warnings for a cleaner terminal 
warnings.filterwarnings("ignore")

class BlipProcessor:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initializes the BLIP model and processor from Hugging Face.
        The model is loaded into memory only ONCE when the class is instantiated.
        """
        print(f"Loading BLIP model '{model_name}'. This might take a moment...")
        self.processor = HuggingFaceBlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        print("BlipProcessor initialized successfully.")

    def process(self, image_path: str, max_new_tokens: int = 40) -> str:
        """
        Processes a single image and returns a generated text caption.
        Example output: "a group of people playing badminton on a court"
        """
        if not os.path.exists(image_path):
             print(f"Image not found: {image_path}")
             return ""

        try:
            # 1. Load and prepare the image securely
            raw_image = Image.open(image_path).convert('RGB')
            
            # 2. Preprocess image for the model (convert to tensors)
            inputs = self.processor(raw_image, return_tensors="pt")
            
            # 3. Generate caption with a token limit for speed optimization
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # 4. Decode the output tokens into a readable human string
            generated_caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return generated_caption.strip()
            
        except Exception as e:
            print(f"Error processing caption for {image_path}: {e}")
            return "" 