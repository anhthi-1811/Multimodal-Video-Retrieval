"""
=============================================================================
ASR PROCESSOR (FEATURE EXTRACTION)
=============================================================================
Description:
Contains the AsrProcessor class. Responsible ONLY for initializing the 
Faster-Whisper model and transcribing audio from a single video/audio file.
Does not handle database connections, multiprocessing, or file system loops.
=============================================================================
"""

import os
import warnings
from faster_whisper import WhisperModel

# Suppress unnecessary warning logs for a cleaner terminal
warnings.filterwarnings("ignore")

class AsrProcessor:
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the Faster-Whisper model.
        The model is loaded into memory only ONCE when the class is instantiated.
        Optimized for CPU usage by default.
        """
        # Load the model with specified optimizations
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"AsrProcessor initialized (Model: {model_size}, Device: {device}, Compute: {compute_type})")

    def process(self, file_path: str, lang: str = "vi") -> list:
        """
        Processes a single video/audio file and returns a list of dictionaries 
        containing the transcribed text along with their timestamps.
        Example output: [{'start': 0.0, 'end': 2.5, 'text': 'Xin chào'}]
        """
        if not os.path.exists(file_path):
             print(f"Media file not found: {file_path}")
             return []

        try:
            # Execute Whisper to extract audio transcript and timestamps
            segments, info = self.model.transcribe(file_path, beam_size=5, language=lang)
            
            transcript_data = []
            
            # Loop through each spoken segment and extract start, end, and text
            for segment in segments:
                transcript_data.append({
                    "start": round(segment.start, 2),  # Round to 2 decimal places
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })
            
            return transcript_data
            
        except Exception as e:
            print(f"Error processing ASR for {file_path}: {e}")
            return []