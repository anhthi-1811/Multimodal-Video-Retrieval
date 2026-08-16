"""
=============================================================================
AUDIO INGESTION PIPELINE 
=============================================================================
Description:
This script scans the 'Videos' directory for .mp4 files and uses the 
AsrProcessor to transcribe the audio. The transcriptions are then pushed 
to a dedicated 'video_data' collection in MongoDB Atlas.
=============================================================================
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# 1. PATH CONFIGURATION
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.database import MongoManager
from src.feature_extraction import AsrProcessor

load_dotenv(os.path.join(project_root, '.env'))
MONGO_URI = os.getenv("MONGO_URI")

VIDEO_DIR = os.path.join(project_root, 'data', 'Videos')
JSON_OUTPUT_DIR = os.path.join(project_root, 'data', 'processed_data')
JSON_FILE_PATH = os.path.join(JSON_OUTPUT_DIR, 'asr_results.json')

def main(): 
    print("==================================================")
    print("STARTING AUDIO INGESTION PIPELINE (ASR)") 
    print("==================================================\n")

    # 2. INITIALIZATION
    print("Initializing Whisper Model and Database...")
    db = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='video_data')
    asr = AsrProcessor(model_size="small", device="cpu", compute_type="int8")
    
    # 3. SCAN DIRECTORY & RESUME MECHANISM
    all_videos = []
    for dirpath, _, filenames in os.walk(VIDEO_DIR):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                all_videos.append(os.path.join(dirpath, filename))
                
    # Check Mongo for existing transcripts
    existing_docs = db.collection.find({}, {"video_id": 1})
    processed_ids = set(doc["video_id"] for doc in existing_docs if "video_id" in doc)
    
    pending_videos = []
    for video_path in all_videos:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if video_id not in processed_ids:
            pending_videos.append(video_path)

    print(f"Found {len(all_videos)} videos. {len(processed_ids)} already processed.")
    print(f"Remaining workload: {len(pending_videos)} videos to transcribe.\n")

    # 4. TRANSCRIBE & PUSH
    start_time = time.time()
    all_results = []
    
    for index, video_path in enumerate(pending_videos, start=1):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{index}/{len(pending_videos)}] Listening to: {video_id}...")
        
        # transcript_data is now a list of dictionaries with timestamps
        transcript_data = asr.process(video_path, lang="vi")
        
        # Combine all segments into a single full text string for easy preview/search
        full_text = " ".join([seg["text"] for seg in transcript_data])
        
        payload = {
            "video_id": video_id,
            "full_asr_text": full_text,     # The combined string
            "segments": transcript_data     # The detailed list with timestamps
        }
        
        db.upsert_frame_data(video_id, payload, id_field="video_id")
        all_results.append(payload)
        
        # Preview the extracted text in the console
        preview = full_text[:50] + "..." if len(full_text) > 50 else full_text
        print(f"Text: {preview if preview else 'No speech detected'}\n") 

    # 5. BACKUP & CLEANUP
    if all_results:
        os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
        with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
            
    db.close()
    print(f"ASR PIPELINE COMPLETED in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main() 