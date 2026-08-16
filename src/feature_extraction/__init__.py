"""
=============================================================================
FEATURE EXTRACTION PACKAGE 
=============================================================================
Description:
Contains AI processor classes responsible for extracting features from 
raw data (images and audio). Each processor is designed to load its 
heavy model once during initialization.
=============================================================================
"""


from .yolo_processor import YoloProcessor
from .ocr_processor import OcrProcessor
from .blip_processor import BlipProcessor
from .asr_processor import AsrProcessor

__all__ = [
    'YoloProcessor',
    'OcrProcessor',
    'BlipProcessor', 
    'AsrProcessor'
]  