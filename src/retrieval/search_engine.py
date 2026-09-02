"""
=============================================================================
SEARCH ENGINE (CORE RETRIEVAL OPERATIONS)
=============================================================================
Description:
This module acts as the underlying execution engine (the "muscle") for the 
retrieval system. It initializes the embedding models and FAISS index spaces.
It exposes atomic, modality-specific search functions (Visual, OCR, ASR) 
that take a query, encode it, perform vector search, and return raw similarity 
scores without making any dynamic routing decisions.
=============================================================================
"""

import os
import sys 
import math 
from collections import defaultdict

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(project_root)

from src.vector_search.embedding_models import TextEncoder, CLIPEncoder
from src.vector_search.faiss_manager import FaissManager

class SearchEngine:
    def __init__(self, vector_db_dir: str):
        """
        Initializes the AI Encoders and loads all FAISS index spaces into RAM.
        """
        print("Initializing Search Engine (Decoupled Mode)...")
        
        # 1. Load Encoders (Models) 
        self.text_encoder = TextEncoder(model_name='BAAI/bge-m3')
        self.clip_encoder = CLIPEncoder(model_name='openai/clip-vit-base-patch32')
        
        # 2. Load FAISS Managers
        self.faiss_ocr = FaissManager(
            index_path=os.path.join(vector_db_dir, 'ocr_text.index'),
            map_path=os.path.join(vector_db_dir, 'ocr_text_map.json'),
            dimension=1024
        )
        self.faiss_caption = FaissManager(
            index_path=os.path.join(vector_db_dir, 'blip_caption.index'),
            map_path=os.path.join(vector_db_dir, 'blip_caption_map.json'),
            dimension=1024
        )
        self.faiss_asr = FaissManager(
            index_path=os.path.join(vector_db_dir, 'asr_audio.index'),
            map_path=os.path.join(vector_db_dir, 'asr_audio_map.json'),
            dimension=1024
        )
        self.faiss_yolo = FaissManager(
            index_path=os.path.join(vector_db_dir, 'yolo_objects.index'),
            map_path=os.path.join(vector_db_dir, 'yolo_objects_map.json'),
            dimension=512
        )
        self.faiss_image = FaissManager(
            index_path=os.path.join(vector_db_dir, 'image_raw.index'),
            map_path=os.path.join(vector_db_dir, 'image_raw_map.json'),
            dimension=512
        )
        print("Search Engine is ready!")

    def _expand_asr_segment(self, segment_id: str) -> list:
        """
        Maps an ASR segment ID to a list of physical Frame IDs.
        Example: "L21_V001_18.5_20.0" -> ["L21_V001_018", "L21_V001_019", "L21_V001_020"]
        """
        try:
            parts = segment_id.split('_')
            video_id = f"{parts[0]}_{parts[1]}"
            start_time = float(parts[2])
            end_time = float(parts[3])
            
            start_frame = math.floor(start_time)
            if start_frame == 0: 
                start_frame = 1 
                
            end_frame = math.ceil(end_time)
            
            frame_ids = []
            for i in range(start_frame, end_frame + 1):
                frame_ids.append(f"{video_id}_{i:03d}")
                
            return frame_ids
            
        except Exception as e:
            print(f"Error parsing segment ID {segment_id}: {e}")
            return []

    def search_visual(self, query: str, top_k: int = 20) -> dict:
        """
        Executes a dense vector search across Image, YOLO, and Caption indices.
        Returns a dictionary of {frame_id: aggregated_score}.
        """
        score_ledger = defaultdict(float)
        
        vec_clip = self.clip_encoder.encode_text(query)    # 512D
        vec_bge = self.text_encoder.encode(query)          # 1024D
        
        img_ids, img_scores = self.faiss_image.search(vec_clip, top_k=top_k)
        yolo_ids, yolo_scores = self.faiss_yolo.search(vec_clip, top_k=top_k)
        cap_ids, cap_scores = self.faiss_caption.search(vec_bge, top_k=top_k)
        
        # Base Late Fusion for Visual Modality
        for fid, score in zip(img_ids, img_scores):
            score_ledger[fid] += score 
        for fid, score in zip(yolo_ids, yolo_scores):
            score_ledger[fid] += score 
        for fid, score in zip(cap_ids, cap_scores): 
            score_ledger[fid] += score 

        return dict(sorted(score_ledger.items(), key=lambda item: item[1], reverse=True)[:top_k])

    def search_ocr(self, query: str, top_k: int = 20) -> dict:
        """
        Executes a dense vector search strictly on the OCR index.
        Returns a dictionary of {frame_id: score}.
        """
        score_ledger = defaultdict(float)
        vec_bge_ocr = self.text_encoder.encode(query) 
        ocr_ids, ocr_scores = self.faiss_ocr.search(vec_bge_ocr, top_k=top_k)
        
        for fid, score in zip(ocr_ids, ocr_scores):
            score_ledger[fid] += score
            
        return dict(sorted(score_ledger.items(), key=lambda item: item[1], reverse=True)[:top_k])

    def search_asr(self, query: str, top_k: int = 20) -> dict:
        """
        Executes a dense vector search on the ASR index and maps segments back to frames.
        Returns a dictionary of {frame_id: score}.
        """
        score_ledger = defaultdict(float)
        vec_bge_asr = self.text_encoder.encode(query)
        asr_ids, asr_scores = self.faiss_asr.search(vec_bge_asr, top_k=top_k)
        
        for seg_id, score in zip(asr_ids, asr_scores):
            mapped_frame_ids = self._expand_asr_segment(seg_id)
            for fid in mapped_frame_ids:
                score_ledger[fid] += score
                
        return dict(sorted(score_ledger.items(), key=lambda item: item[1], reverse=True)[:top_k]) 