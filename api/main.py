"""
=============================================================================
BACKEND API SERVER (FASTAPI)
=============================================================================
Description:
Acts as the bridge between the UI and the Core Engine. It receives HTTP 
requests, orchestrates the LLM Agent and Search Engine, enriches results 
with MongoDB metadata, and returns a structured JSON response.
=============================================================================
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Setup paths to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.database import MongoManager
from src.retrieval.search_engine import SearchEngine
from src.retrieval.query_agent import QueryAgent

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))
MONGO_URI = os.getenv("MONGO_URI")

# Define base directories for image paths
KEYFRAMES_DIR = os.path.join(project_root, 'data', 'keyframes')
VECTOR_DB_DIR = os.path.join(project_root, 'data', 'vector_db')

# ---------------------------------------------------------------------------
# 1. INIT FASTAPI & GLOBAL MODULES
# ---------------------------------------------------------------------------
app = FastAPI(title="AIC 2026 Multimodal Search API", version="1.0")

# Global instances (Loaded once when server starts)
db_visual = None
search_engine = None
query_agent = None

@app.on_event("startup")
def startup_event():
    """Initializes heavy models and database connections at server startup."""
    global db_visual, search_engine, query_agent
    print("[API] Starting up server, loading AI models...")
    
    db_visual = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='keyframes_data')
    query_agent = QueryAgent()
    search_engine = SearchEngine(vector_db_dir=VECTOR_DB_DIR)
    
    print("[API] All systems ready!")

@app.on_event("shutdown")
def shutdown_event():
    if db_visual:
        db_visual.close()
    print("[API] Server shut down gracefully.")

# ---------------------------------------------------------------------------
# 2. DATA MODELS (PYDANTIC)
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_image_path(frame_id: str) -> str:
    """Reconstructs the absolute path to the physical image file."""
    try:
        parts = frame_id.rsplit('_', 1)
        if len(parts) == 2:
            batch_num = parts[0].split('_')[0].replace('L', '')
            batch_folder = f"Keyframes_L{batch_num}"
            return os.path.join(KEYFRAMES_DIR, batch_folder, "keyframes", parts[0], f"{parts[1]}.jpg")
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
# 4. API ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/search")
async def perform_search(request: SearchRequest):
    """
    Main endpoint for multimodal search.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    start_time = time.time()
    
    try:
        # Step 1: Agent parses the query
        parsed_query = query_agent.parse_query(request.query)
        
        # Step 2: Search Engine retrieves top Frame IDs
        top_results = search_engine.search(parsed_query, top_k=request.top_k)
        
        # Step 3: Enrich with Metadata from MongoDB
        final_results = []
        for rank, (frame_id, score) in enumerate(top_results, start=1):
            doc = db_visual.collection.find_one({"frame_id": frame_id})
            
            # Extract metadata safely
            caption = doc.get("blip_caption", "") if doc else ""
            ocr = doc.get("ocr_text", "") if doc else ""
            yolo = doc.get("yolo_objects", {}) if doc else {}
            
            final_results.append({
                "rank": rank,
                "frame_id": frame_id,
                "score": round(score, 4),
                "image_path": get_image_path(frame_id),
                "metadata": {
                    "caption": caption,
                    "ocr": ocr,
                    "yolo_objects": yolo
                }
            })
            
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "processing_time_sec": round(processing_time, 3),
            "agent_thought_process": parsed_query,
            "results": final_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))