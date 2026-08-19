"""
=============================================================================
VECTOR INDEXING PIPELINE (ORCHESTRATOR)
=============================================================================
Description:
Acts as the central orchestrator. It fetches raw metadata from MongoDB,
passes them through the Encoders to generate high-dimensional vectors, 
and stores these vectors into 5 independent FAISS indices.
=============================================================================
"""

import os
import sys
import time 
from dotenv import load_dotenv

# 1. PATH CONFIGURATION
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)               

from src.database import MongoManager
from src.vector_search.embedding_models import TextEncoder, CLIPEncoder 
from src.vector_search.faiss_manager import FaissManager 

# Load environment variables 
load_dotenv(os.path.join(project_root, '.env')) 
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("Missing MONGO_URI. Please check your .env file.")

# Define Data Directories 
DATA_ROOT = os.path.join(project_root, 'data')  
KEYFRAMES_DIR = os.path.join(DATA_ROOT, 'keyframes') 
VECTOR_DB_DIR = os.path.join(DATA_ROOT, 'vector_db')

def main():
    print("==================================================")
    print("STARTING VECTOR INDEXING PIPELINE")
    print("==================================================\n") 

    # -----------------------------------------------------------------------
    # PHASE 1: INITIALIZATION
    # -----------------------------------------------------------------------
    print("[1/4] Initializing Database, Encoders, and FAISS Managers...")
    
    db_visual = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='keyframes_data')
    db_audio = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='video_data')
    
    text_encoder = TextEncoder(model_name='BAAI/bge-m3') 
    clip_encoder = CLIPEncoder(model_name='openai/clip-vit-base-patch32') 
    
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    faiss_image = FaissManager(index_path=os.path.join(VECTOR_DB_DIR, 'image_raw.index'), map_path=os.path.join(VECTOR_DB_DIR, 'image_raw_map.json'), dimension=512)
    faiss_yolo = FaissManager(index_path=os.path.join(VECTOR_DB_DIR, 'yolo_objects.index'), map_path=os.path.join(VECTOR_DB_DIR, 'yolo_objects_map.json'), dimension=512)
    faiss_ocr = FaissManager(index_path=os.path.join(VECTOR_DB_DIR, 'ocr_text.index'), map_path=os.path.join(VECTOR_DB_DIR, 'ocr_text_map.json'), dimension=1024)
    faiss_caption = FaissManager(index_path=os.path.join(VECTOR_DB_DIR, 'blip_caption.index'), map_path=os.path.join(VECTOR_DB_DIR, 'blip_caption_map.json'), dimension=1024)
    faiss_asr = FaissManager(index_path=os.path.join(VECTOR_DB_DIR, 'asr_audio.index'), map_path=os.path.join(VECTOR_DB_DIR, 'asr_audio_map.json'), dimension=1024)

    start_process = time.time()

    try:
        # -----------------------------------------------------------------------
        # PHASE 2: VISUAL DATA PIPELINE (OCR, Caption, YOLO, Image)
        # -----------------------------------------------------------------------
        print("\n[2/4] Fetching and Processing VISUAL DATA (keyframes_data)...")
        visual_docs = list(db_visual.collection.find({}))
        total_visual = len(visual_docs)
        
        for index, doc in enumerate(visual_docs, start=1):
            frame_id = doc.get("frame_id")
            print(f"  -> Processing Visual [{index}/{total_visual}]: {frame_id}")
            
            parts = frame_id.rsplit('_', 1)
            if len(parts) == 2:
                batch_num = parts[0].split('_')[0].replace('L', '') 
                batch_folder = f"Keyframes_L{batch_num}"
                image_path = os.path.join(KEYFRAMES_DIR, batch_folder, "keyframes", parts[0], f"{parts[1]}.jpg")
            else:
                image_path = ""
                
            ocr_text = doc.get("ocr_text", "")
            if ocr_text: 
                faiss_ocr.add(text_encoder.encode(ocr_text), frame_id)
                
            blip_caption = doc.get("blip_caption", "")
            if blip_caption:
                faiss_caption.add(text_encoder.encode(blip_caption), frame_id)
                
            yolo_dict = doc.get("yolo_objects", {})
            if yolo_dict:
                object_text = ", ".join([f"{count} {obj}" for obj, count in yolo_dict.items()])
                faiss_yolo.add(clip_encoder.encode_text(object_text), frame_id)
                
            # Phần Image gốc:
            if os.path.exists(image_path):
                faiss_image.add(clip_encoder.encode_image(image_path), frame_id)
            else:
                print(f"     [WARNING] Image not found: {image_path}") 

        # -----------------------------------------------------------------------
        # PHASE 3: AUDIO DATA PIPELINE (ASR)
        # -----------------------------------------------------------------------
        print("\n[3/4] Fetching and Processing AUDIO DATA (video_data)...")
        audio_docs = list(db_audio.collection.find({}))
        total_audio = len(audio_docs)
        
        for index, doc in enumerate(audio_docs, start=1):
            video_id = doc.get("video_id")
            segments = doc.get("segments", [])
            print(f"  -> Processing Audio [{index}/{total_audio}]: {video_id} ({len(segments)} segments)")
            
            for seg in segments:
                text = seg.get("text", "")
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                
                if text:
                    segment_id = f"{video_id}_{start}_{end}"
                    faiss_asr.add(text_encoder.encode(text), segment_id)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Bạn vừa chủ động ngắt tiến trình (Ctrl+C)!")
        print("Đang tiến hành lưu lại toàn bộ dữ liệu đã xử lý được xuống ổ cứng để tránh mất mát...")
        
    except Exception as e:
        print(f"\n\n[ERROR] Đã xảy ra lỗi bất ngờ: {e}")
        print("Đang tiến hành lưu lại dữ liệu để bảo toàn kết quả...") 

    finally:
        # -----------------------------------------------------------------------
        # PHASE 4: PERSIST TO DISK & CLEANUP
        # -----------------------------------------------------------------------
        print("\n[4/4] Saving all FAISS indices to disk...")
        start_save = time.time()
        
        faiss_image.save()
        faiss_yolo.save()
        faiss_ocr.save()
        faiss_caption.save()
        faiss_asr.save()
        
        db_visual.close()
        db_audio.close()
        
        print(f"\nALL DONE! Vector Indexing completed successfully in: {time.time() - start_save:.2f}s")
        print("==================================================")

if __name__ == "__main__":
    main() 