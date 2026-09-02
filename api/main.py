"""
=============================================================================
BACKEND API SERVER (FASTAPI) 
=============================================================================
Description:
Acts as the bridge between the UI and the Core Engine. It receives HTTP 
requests, delegates the complex reasoning and retrieval to the RetrievalAgent,
enriches results with MongoDB metadata, and returns a structured JSON response.
=============================================================================
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import List, Optional
from dotenv import load_dotenv

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

load_dotenv(os.path.join(project_root, '.env'))
MONGO_URI = os.getenv("MONGO_URI")
# ==========================================

from src.database.mongo_manager import MongoManager
from src.agent.retrieval_agent import RetrievalAgent

# Define base directories for image paths
KEYFRAMES_DIR = os.path.join(project_root, 'data', 'keyframes')

# ---------------------------------------------------------------------------
# 1. INIT FASTAPI & GLOBAL MODULES
# ---------------------------------------------------------------------------
app = FastAPI(title="AIC 2026 AI Agent API", version="2.0")

# Global instances (Loaded once when server starts)
db_visual = None
retrieval_agent = None

@app.on_event("startup")
def startup_event():
    """Initializes heavy models and database connections at server startup."""
    global db_visual, retrieval_agent
    print("[API] Starting up server, loading AI Agent and Models...")
    
    db_visual = MongoManager(uri=MONGO_URI, db_name='aic_2026_db', collection_name='keyframes_data')
    
    # Initialize the new Brain
    retrieval_agent = RetrievalAgent()
    
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
    top_k: Optional[int] = 50

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
    Main endpoint for multimodal Agentic search.
    """ 
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    start_time = time.time()
    
    try:
        # Step 1: Delegate the entire reasoning & searching process to the Agent
        # The Agent will automatically decide which tools to use and evaluate scores
        top_results = retrieval_agent.run(user_query=request.query, score_threshold=0.25)
        
        # Limit to the requested top_k items
        top_results = top_results[:request.top_k]
        
        # Step 2: Enrich with Metadata from MongoDB
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
            "agent_status": "ReAct Loop Completed",
            "results": final_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))