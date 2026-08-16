"""
=============================================================================
YOLO PROCESSOR (FEATURE EXTRACTION)
=============================================================================
Description:
Contains the YoloProcessor class. Responsible ONLY for loading the YOLOv8 
model and extracting object counts from a single image.
Does not handle database connections, multiprocessing, or file system loops.
=============================================================================
"""

import os 
import warnings
from collections import Counter
from ultralytics import YOLO

# Suppress unnecessary warnings for a cleaner terminal
warnings.filterwarnings("ignore")

class YoloProcessor:
    def __init__(self, model_path: str = 'weights/yolov8n.pt'):
        """
        Initializes the YOLOv8 model.
        The model is loaded into memory only ONCE when the class is instantiated.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}. Please place it in the 'weights/' directory.")
        
        # Load lightweight YOLOv8 nano model
        self.model = YOLO(model_path)
        print(f"YoloProcessor initialized with model: {model_path}")

    def process(self, image_path: str) -> dict:
        """
        Processes a single image and returns a dictionary of detected objects.
        Example output: {'person': 2, 'car': 1, 'sports ball': 1}
        """
        if not os.path.exists(image_path):
             print(f"Image not found: {image_path}")
             return {}

        try:
            # Inference (verbose=False to disable spamming the terminal per image)
            inference_results = self.model(image_path, verbose=False)
            result = inference_results[0]
            
            # Map class IDs to string names
            detected_classes = [self.model.names[int(c)] for c in result.boxes.cls]
            
            # Count occurrences of each class
            class_counts = Counter(detected_classes)
            
            # Return as a standard Python dictionary
            return dict(class_counts) if class_counts else {}
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {}