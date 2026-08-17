# AIC 2026 Multimodal Search System

An advanced, highly modular multimodal retrieval system built for the AI Challenge (AIC) 2026. Designed with a **decoupled Micro-Monolith architecture**, the system seamlessly integrates state-of-the-art vision and language models while prioritizing scalability, maintainability, and computational efficiency. 

By leveraging an LLM-powered Query Agent as a semantic router, the system dynamically orchestrates highly accurate searches across video keyframes using visual, audio (ASR), object detection (YOLO), and textual (OCR/Caption) vectors, ensuring high precision (mAP) and optimal resource utilization.

## Architectural Highlights & Key Features

* **Smart Query Routing (LLM Agent):** Utilizes Google Gemini 1.5 strictly via Structured JSON Outputs to parse natural language queries, strip noise, and classify intents. This acts as an intelligent gateway to prevent redundant vector database scans.
* **Selective Encoding & Lazy Searching:** Drastically reduces CPU/Memory overhead. The retrieval engine only embeds and searches modalities explicitly present in the user's parsed prompt (e.g., bypassing audio index searches if no sound-related keywords exist).
* **Dynamic Late Fusion Strategy:** Replaces hard-coded heuristic weights with dynamic, AI-assigned weights per query intent, allowing the system to naturally emphasize specific modalities (like OCR for specific text/logos or ASR for speeches).
* **Decoupled Full-Stack Design:** 
  * **Backend:** Built on **FastAPI**, providing asynchronous, high-throughput RESTful endpoints ready for containerization (Docker).
  * **Frontend:** A lightweight **Streamlit** interface for rapid prototyping, qualitative evaluation, and visual debugging.
  * **Storage:** Hybrid approach combining **FAISS** for fast dense vector similarity search and **MongoDB** for flexible metadata enrichment.

## Technology Stack

* **Core AI Models:** `openai/clip-vit-base-patch32` (Visual), `BAAI/bge-m3` (Text/OCR/ASR).
* **Feature Extraction:** PyTorch, YOLOv8 (Objects), PaddleOCR/EasyOCR (Text), Faster-Whisper (Audio).
* **Vector Search & Storage:** FAISS (CPU), MongoDB Atlas.
* **LLM Engine:** Google Gemini API (`google-generativeai`).
* **Backend:** FastAPI, Uvicorn, Pydantic.
* **Frontend:** Streamlit.

## High-Level Data Flow

1. `Client` sends a natural language query -> `FastAPI Gateway`.
2. `Query Agent (LLM)` decomposes the string into a structured JSON `(Visual, OCR, ASR, Weights)`.
3. `Search Engine` performs Selective Encoding (BGE-M3 / CLIP) based on the JSON.
4. `FAISS Managers` execute KNN similarity searches on the relevant indices.
5. `Late Fusion Module` aggregates and reranks frame scores using dynamic weights.
6. `MongoDB` enriches the Top-K frames with physical paths and metadata.
7. `Response` is delivered to the UI for rendering. 

## Project Structure

```text
AIC_2026/
├── api/
│   └── main.py                  # FastAPI server and endpoints
├── data/                        # Local data (Ignored by Git)
│   ├── keyframes/               # Physical image files
│   └── vector_db/               # FAISS index files (*.index)
├── scripts/
│   ├── evaluate.py              # Script to calculate mAP and Recall@K
│   ├── run_search.py            # CLI search execution
│   └── run_vector_indexing.py   # Vectorization pipeline
├── src/
│   ├── database/
│   │   └── mongo_manager.py     # MongoDB connection and queries
│   ├── retrieval/
│   │   ├── query_agent.py       # LLM intent parsing
│   │   └── search_engine.py     # Dynamic Late Fusion logic
│   └── vector_search/
│       ├── embedding_models.py  # CLIP and BGE model wrappers
│       └── faiss_manager.py     # FAISS indexing and search
├── ui/ 
│   └── web_app.py               # Streamlit web interface
├── .env                         # Environment variables (Ignored by Git)
├── .gitignore                   # Git ignore configurations
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation 
```

## Installation & Setup

### 1. Prerequisites

Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/AIC-2026-Multimodal-Search.git
cd AIC-2026-Multimodal-Search 
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Environment Variables
Create a .env file in the root directory and add your credentials:
```bash
MONGO_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

## How to Run the System
The system requires both the Backend (FastAPI) and the Frontend (Streamlit) to run simultaneously. You will need to open two separate terminal windows.

### Terminal 1: Start the Backend (FastAPI)
This will initialize the AI models, connect to MongoDB, and open the API gateway.
```bash
uvicorn api.main:app --reload --port 8000
```
You can access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

### Terminal 2: Start the Frontend (Streamlit)
This will launch the graphical user interface.
```bash
streamlit run ui/web_app.py
```

The web app will automatically open in your default browser at http://localhost:8501.

## Evaluation
To evaluate the system's performance using standard Information Retrieval metrics (mAP, Recall@K):
   1. Ensure you have `ground_truth.csv` and `my_predictions.csv` inside the data/ folder.
   2. Run the evaluation script: 
   ```bash
   python scripts/evaluate.py
   ``` 
   




