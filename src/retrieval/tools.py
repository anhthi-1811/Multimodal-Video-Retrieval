"""
=============================================================================
AGENT TOOLBOX (FUNCTION CALLING ABSTRACTION)
=============================================================================
Description:
This module defines the callable tools provided to the LLM (Gemini Agent).
Each function is a lightweight wrapper around the core SearchEngine. 
The extensive docstrings serve as system prompts to instruct the LLM on 
exactly WHEN and HOW to use each tool during its reasoning cycle.
=============================================================================
"""

import os
from src.retrieval.search_engine import SearchEngine 

# Initialize the static engine (Load models once into RAM to prevent memory leaks)
vector_db_dir = os.path.join(os.path.dirname(__file__), "../../data/vector_db")
engine = SearchEngine(vector_db_dir=vector_db_dir)

def search_visual_modality(visual_query: str, top_k: int = 20) -> dict:
    """
    Search for video keyframes based strictly on visual descriptions.
    Use this tool when the user's prompt describes:
    - Actions (e.g., "playing badminton", "running", "serving").
    - Objects, colors, or scenes (e.g., "a person in a red shirt", "a stadium").
    
    DO NOT use this tool for finding specific printed text or spoken words.
    The query MUST be translated to English before passing to this tool.
    
    Args:
        visual_query: A clear English description of the visual elements.
        top_k: Number of results to return.
    """
    top_k = int(top_k) 
    print(f"[TOOL LOG] Calling Visual Search with query: '{visual_query}'")
    return engine.search_visual(query=visual_query, top_k=top_k)


def search_text_modality(text_query: str, top_k: int = 20) -> dict:
    """
    Search for video keyframes that contain specific printed text on screen (OCR).
    Use this tool when the user wants to find specific numbers, scoreboards, 
    names printed on jerseys, or sponsor logos.
    
    Args:
        text_query: The exact text, word, or number to search for (e.g., "15", "Yonex").
        top_k: Number of results to return.
    """
    top_k = int(top_k) 
    print(f"[TOOL LOG] Calling OCR Search with query: '{text_query}'")
    return engine.search_ocr(query=text_query, top_k=top_k)


def search_audio_modality(spoken_query: str, top_k: int = 20) -> dict:
    """
    Search for video keyframes based on spoken audio or commentary (ASR).
    Use this tool ONLY when the user explicitly mentions "hearing", 
    "commentator says", or asks for specific spoken dialogues.
    
    Args:
        spoken_query: The spoken phrase or dialogue to search for.
        top_k: Number of results to return.
    """
    top_k = int(top_k) 
    print(f"[TOOL LOG] Calling Audio/ASR Search with query: '{spoken_query}'")
    return engine.search_asr(query=spoken_query, top_k=top_k) 