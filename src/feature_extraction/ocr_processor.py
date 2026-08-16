"""
=============================================================================
OCR PROCESSOR (FEATURE EXTRACTION)
=============================================================================
Description:
Contains the OcrProcessor class. Responsible ONLY for initializing the 
PaddleOCR engine and extracting text from a single image.
Does not handle database connections, multiprocessing, or file system loops.
=============================================================================
"""

import os
import logging
import warnings
from paddleocr import PaddleOCR

# Suppress unnecessary warnings and PaddleOCR's verbose logging
logging.getLogger("ppocr").setLevel(logging.ERROR) 
warnings.filterwarnings("ignore") 

class OcrProcessor:
    def __init__(self, lang: str = 'vi'): 
        """
        Initializes the PaddleOCR engine.
        The model is loaded into memory only ONCE when the class is instantiated.
        """
        # show_log=False prevents PaddleOCR from spamming the terminal during init
        self.ocr = PaddleOCR(
            use_textline_orientation=True, 
            lang=lang, 
            enable_mkldnn=False  
        )
        print(f"OcrProcessor initialized (Language: {lang})")

    def process(self, image_path: str) -> str:
        """
        Processes a single image and returns a single comma-separated string 
        of all detected text.
        Example output: "OLYMFIT, Trận đấu bắt đầu, 2026"
        """
        if not os.path.exists(image_path):
             print(f"Image not found: {image_path}")
             return ""

        try:
            # 1. AI Engine extracts text from the image
            result = self.ocr.ocr(image_path) 
            
            # Check if result is empty or None
            if not result or not result[0]:
                return ""
            
            # 2. Extract text from PaddleOCR's specific output format
            # Format is typically: [[[box_coords], ('text', confidence_score)], ...]
            texts = [line[1][0] for line in result[0] if isinstance(line, list) and len(line) == 2]
            
            # 3. Join isolated text detections into a single comma-separated string
            return ", ".join(texts)
            
        except Exception as e:
            print(f"Error processing OCR for {image_path}: {e}")
            return ""