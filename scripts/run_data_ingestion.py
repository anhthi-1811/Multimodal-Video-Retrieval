"""
=============================================================================
DATA INGESTION PIPELINE
=============================================================================
Description:
This script acts as the main orchestrator for Phase Extraction.
It sequentially integrates YOLO, OCR, and BLIP processors, extracts features  
from all keyframes, pushes the unified data to MongoDB Atlas, and saves
a local JSON backup.
=============================================================================
"""

import os
import sys
import json 
import time
from dotenv import load_dotenv

# --------------------------------------------------------------------------- 
# 1. PATH CONFIGURATION & MODULE IMPORT
# ---------------------------------------------------------------------------
# Dynamically append the project root to sys.path so Python can find 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now we can cleanly import our OOP classes using the __init__.py facades
from src.database import MongoManager
from src.feature_extraction import YoloProcessor, OcrProcessor, BlipProcessor

# Load environment variables (MONGO_URI)
load_dotenv(os.path.join(project_root, '.env'))
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("Missing MONGO_URI. Please check your .env file.")

# Define physical paths
KEYFRAMES_DIR = os.path.join(project_root, 'data', 'keyframes')
JSON_OUTPUT_DIR = os.path.join(project_root, 'data', 'processed_data')
JSON_FILE_PATH = os.path.join(JSON_OUTPUT_DIR, 'phase_a_results.json')

def main():
    print("==================================================")
    print("STARTING DATA INGESTION PIPELINE")
    print("==================================================\n")

    # -----------------------------------------------------------------------
    # 2. INITIALIZE TOOLS (Load Models & DB Connection ONLY ONCE)
    # -----------------------------------------------------------------------
    print("[1/4] INITIALIZING AI MODELS AND DATABASE...")

    db = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='keyframes_data')

    # Instantiate processors (Heavy models are loaded into RAM here)
    yolo_processor = YoloProcessor(model_path=os.path.join(project_root, 'weights', 'yolov8n.pt'))
    ocr_processor = OcrProcessor(lang='vi')
    blip_processor = BlipProcessor()

    print("All systems initialized successfully.\n")

    # -----------------------------------------------------------------------
    # 3. SCAN DIRECTORY & FILTER PROCESSED ONES
    # -----------------------------------------------------------------------
    print("[2/4] SCANNING DATA DIRECTORY...") 
    all_image_paths = []  

    # Traverse directory to discover all valid image files
    for dirpath, _, filenames in os.walk(KEYFRAMES_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
                all_image_paths.append(os.path.join(dirpath, filename))

    print(f"Discovered a total of {len(all_image_paths)} images on local disk.")

    # --- RESUME MECHANISM: QUERY EXISTING FRAMES FROM MONGODB ATLAS ---
    print("Querying MongoDB to fetch existing frame IDs...")
    
    # Projection query: Fetch only 'frame_id' to minimize network overhead and RAM usage
    existing_docs = db.collection.find({}, {"frame_id": 1})
    processed_ids = set(doc["frame_id"] for doc in existing_docs if "frame_id" in doc)
    print(f"Found {len(processed_ids)} frames already indexed in MongoDB.")

    # Filter out images that have already been processed
    pending_image_paths = []
    for img_path in all_image_paths:
        filename = os.path.basename(img_path)
        folder_name = os.path.basename(os.path.dirname(img_path))
        frame_num = os.path.splitext(filename)[0]
        frame_id = f"{folder_name}_{frame_num}"

        # Only stage for inference if the frame does not exist in the database
        if frame_id not in processed_ids:
            pending_image_paths.append(img_path)

    total_images = len(pending_image_paths)
    print(f"Remaining workload: {total_images} images to process!\n")

    # -----------------------------------------------------------------------
    # 4. THE INGESTION LOOP (Extract -> Unified Payload -> Database)
    # -----------------------------------------------------------------------
    print("[3/4] STARTING FEATURE EXTRACTION & CLOUD SYNC...")

    all_results_backup = []
    start_time = time.time()

    for index, img_path in enumerate(pending_image_paths, start=1):
        try:
            # Extract frame_id (e.g., from "L21_V001/020.jpg" -> "L21_V001_020")
            filename = os.path.basename(img_path)
            folder_name = os.path.basename(os.path.dirname(img_path))
            frame_num = os.path.splitext(filename)[0]
            frame_id = f"{folder_name}_{frame_num}"

            # --- EXECUTE AI PROCESSORS ---
            yolo_objects = yolo_processor.process(img_path)
            ocr_text = ocr_processor.process(img_path)
            blip_caption = blip_processor.process(img_path)

            # --- CONSTRUCT UNIFIED DOCUMENT ---
            # MongoDB: grouping all multimodal data together
            payload = {
                "frame_id": frame_id,
                "yolo_objects": yolo_objects,   # dict: {'person': 2, 'racket': 1}
                "ocr_text": ocr_text,           # string: "OLYMFIT, 2026"
                "blip_caption": blip_caption    # string: "a man playing badminton"
            }

            # --- DATABASE UPSERT ---
            db.upsert_frame_data(frame_id, payload)

            # Append to local list for JSON backup
            all_results_backup.append(payload)

            # Console logging 
            yolo_str = ", ".join([f"{k}: {v}" for k, v in yolo_objects.items()]) if yolo_objects else "None"
            
            print(f"[{index}/{total_images}] Processed: {frame_id} | YOLO: [{yolo_str}] | OCR: {bool(ocr_text)} | BLIP: {bool(blip_caption)}")
            
        except Exception as e:
            print(f"Critical error at {img_path}: {e}")

    # -----------------------------------------------------------------------
    # 5. EXPORT BACKUP & CLEANUP
    # -----------------------------------------------------------------------
    print("\n[4/4] EXPORTING LOCAL BACKUP AND CLOSING CONNECTIONS...")

    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results_backup, f, ensure_ascii=False, indent=4)

    db.close()

    elapsed_time = time.time() - start_time
    print(f"PIPELINE COMPLETED in {elapsed_time:.2f} seconds!")
    print("Data pushed to MongoDB Atlas.")
    print(f"Local JSON Backup saved at: {JSON_FILE_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()  