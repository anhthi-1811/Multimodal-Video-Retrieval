"""
=============================================================================
INTERACTIVE SEARCH CLI (PRESENTATION LAYER)
=============================================================================
Description:
Provides a Command Line Interface (CLI) for users to interact with the 
Multimodal Search Engine. It takes the user's text query, retrieves the 
Top K Frame IDs, fetches their metadata from MongoDB, and displays the 
results beautifully on the terminal.
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

# Import internal modules based on the established directory structure
from src.database import MongoManager  # Adjust to src.database.mongo_manager if necessary
from src.retrieval.search_engine import SearchEngine

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))
MONGO_URI = os.getenv("MONGO_URI")

# Define base directories
VECTOR_DB_DIR = os.path.join(project_root, 'data', 'vector_db')
KEYFRAMES_DIR = os.path.join(project_root, 'data', 'keyframes')

def get_image_path(frame_id: str) -> str:
    """
    Reconstructs the physical file path based on the frame_id.
    Example: 'L21_V001_020' -> '.../data/keyframes/Keyframes_L21/keyframes/L21_V001/020.jpg'
    """
    try:
        parts = frame_id.rsplit('_', 1)
        if len(parts) == 2:
            batch_num = parts[0].split('_')[0].replace('L', '') # Extracts "21"
            batch_folder = f"Keyframes_L{batch_num}"
            return os.path.join(KEYFRAMES_DIR, batch_folder, "keyframes", parts[0], f"{parts[1]}.jpg")
    except Exception:
        pass
    return "Path unknown"

def main():
    print("==================================================")
    print("   AIC 2026 - MULTIMODAL SEARCH ENGINE TERMINAL   ")
    print("==================================================\n")

    # -----------------------------------------------------------------------
    # PHASE 1: WARM-UP & INITIALIZATION
    # -----------------------------------------------------------------------
    print("[SYSTEM] Booting up database connections and AI models...")
    start_boot = time.time()
    
    # Connect to MongoDB to fetch metadata later
    db_visual = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='keyframes_data')
    
    # Initialize the core Search Engine (loads FAISS and Encoders into RAM)
    search_engine = SearchEngine(vector_db_dir=VECTOR_DB_DIR)
    
    print(f"[SYSTEM] Ready in {time.time() - start_boot:.2f} seconds.\n")
    print("Type 'exit' or 'quit' at any time to stop the program.")
    print("-" * 50)

    # -----------------------------------------------------------------------
    # PHASE 2: INTERACTIVE LOOP
    # -----------------------------------------------------------------------
    while True:
        try:
            # 2.1 Get user input
            query = input("\n[USER] Nhập câu hỏi truy vấn: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("\n[SYSTEM] Shutting down. Goodbye!")
                break
                
            if not query:
                continue

            # 2.2 Execute Search
            print("[SYSTEM] Đang tìm kiếm đa phương thức...")
            start_search = time.time()
            
            # Change top_k if you want more or fewer results
            top_results = search_engine.search(query, top_k=5)
            
            search_time = time.time() - start_search

            # 2.3 Process and Display Results
            print(f"\n==================================================")
            print(f" KẾT QUẢ CHO: \"{query}\" (Mất {search_time:.3f}s)")
            print(f"==================================================")
            
            if not top_results:
                print("Không tìm thấy kết quả nào phù hợp.")
                continue

            # Loop through the Top K results
            for rank, (frame_id, total_score) in enumerate(top_results, start=1):
                # Fetch metadata from MongoDB
                doc = db_visual.collection.find_one({"frame_id": frame_id})
                
                # Format variables for display
                caption = doc.get("blip_caption", "N/A") if doc else "N/A"
                ocr = doc.get("ocr_text", "N/A") if doc else "N/A"
                yolo = doc.get("yolo_objects", {}) if doc else {}
                
                yolo_str = ", ".join([f"{k}({v})" for k, v in yolo.items()]) if yolo else "N/A"
                img_path = get_image_path(frame_id)

                # Print to terminal
                print(f"[TOP {rank}] - Khung hình: {frame_id} | Điểm tổng: {total_score:.3f}")
                print(f"   > Hình ảnh : {img_path}")
                print(f"   > Caption  : {caption}")
                print(f"   > OCR Text : {ocr}")
                print(f"   > Objects  : {yolo_str}")
                print("-" * 50)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n[SYSTEM] Interrupted by user. Shutting down...")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")

    # Cleanup
    db_visual.close()

if __name__ == "__main__":
    main() 