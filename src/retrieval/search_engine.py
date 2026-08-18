"""
=============================================================================
SEARCH ENGINE (RETRIEVAL & DYNAMIC LATE FUSION)
=============================================================================
Description:
Acts as the central query processor. It receives a parsed JSON query from 
the LLM Agent, selectively encodes only the necessary components, searches 
across relevant FAISS indices, maps ASR segments back to physical frames, 
and performs Dynamic Late Fusion based on AI-assigned weights.
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
        Initializes the AI Encoders and loads all 5 FAISS index spaces into RAM.
        """
        print("Initializing Search Engine (Dynamic Fusion Mode)...")
        
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
        Helper method to map an ASR segment ID to a list of physical Frame IDs.
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

    def search(self, parsed_query: dict, top_k: int = 15) -> list:
        """
        Executes a Multimodal Search based on a parsed dictionary.
        Returns a sorted list of tuples: [(frame_id, total_score), ...]
        """
        if not parsed_query:
            return []

        # 1. EXTRACT DATA FROM AGENT'S PARSED JSON
        visual_query = parsed_query.get("visual_query", "").strip()
        ocr_query = parsed_query.get("ocr_query", "").strip()
        asr_query = parsed_query.get("asr_query", "").strip()
        
        weights = parsed_query.get("weights", {
            "image": 1.0, "caption": 1.0, "yolo": 1.0, "ocr": 1.0, "asr": 1.0
        })

        score_ledger = defaultdict(float)

        # ---------------------------------------------------------
        # 2. LAZY SEARCHING & SELECTIVE ENCODING
        # ---------------------------------------------------------

        # A. VISUAL MODALITY (Searches Image, YOLO, and Caption)
        if visual_query: 
            vec_clip = self.clip_encoder.encode_text(visual_query)    # 512D
            vec_bge_visual = self.text_encoder.encode(visual_query)   # 1024D
            
            # Search
            img_ids, img_scores = self.faiss_image.search(vec_clip, top_k=top_k)
            yolo_ids, yolo_scores = self.faiss_yolo.search(vec_clip, top_k=top_k)
            cap_ids, cap_scores = self.faiss_caption.search(vec_bge_visual, top_k=top_k)
            
            # Fuse Scores
            for fid, score in zip(img_ids, img_scores):
                score_ledger[fid] += score * weights.get('image', 1.0)
            for fid, score in zip(yolo_ids, yolo_scores):
                score_ledger[fid] += score * weights.get('yolo', 1.0)
            for fid, score in zip(cap_ids, cap_scores):
                score_ledger[fid] += score * weights.get('caption', 1.0)

        # B. OCR MODALITY
        if ocr_query:
            vec_bge_ocr = self.text_encoder.encode(ocr_query) # 1024D
            ocr_ids, ocr_scores = self.faiss_ocr.search(vec_bge_ocr, top_k=top_k)
            
            for fid, score in zip(ocr_ids, ocr_scores):
                score_ledger[fid] += score * weights.get('ocr', 1.0)

        # C. ASR MODALITY
        if asr_query:
            vec_bge_asr = self.text_encoder.encode(asr_query) # 1024D
            asr_ids, asr_scores = self.faiss_asr.search(vec_bge_asr, top_k=top_k)
            
            for seg_id, score in zip(asr_ids, asr_scores):
                mapped_frame_ids = self._expand_asr_segment(seg_id)
                for fid in mapped_frame_ids:
                    score_ledger[fid] += score * weights.get('asr', 1.0)

        # ---------------------------------------------------------
        # 3. RERANKING
        # ---------------------------------------------------------
        reranked_results = sorted(score_ledger.items(), key=lambda item: item[1], reverse=True)
        return reranked_results[:top_k] 