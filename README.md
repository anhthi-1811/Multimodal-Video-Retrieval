# AIC 2026 Multimodal Search System

An advanced, highly modular multimodal retrieval system built for the AI Challenge (AIC) 2026. Designed with a **decoupled Micro-Monolith architecture**, the system seamlessly integrates state-of-the-art vision and language models while prioritizing scalability, maintainability, and computational efficiency. 

By leveraging an autonomous **ReAct (Reasoning and Acting) Agent** powered by Gemini Tool Calling, the system dynamically orchestrates highly accurate searches across video keyframes using visual, audio (ASR), object detection (YOLO), and textual (OCR/Caption) vectors, ensuring high precision (mAP) and optimal resource utilization.

## Architectural Highlights & Key Features

* **Autonomous Agentic Routing (Tool Calling):** Utilizes Google Gemini 3.6 via native Tool Calling to create an autonomous ReAct loop. The agent understands user intent, breaks down complex queries, and dynamically invokes specific search modalities (Visual, Text, Audio) without hard-coded logic.
* **Iterative Threshold Evaluation:** The Agent acts as an intelligent orchestrator, evaluating retrieval scores after each tool execution. It continues to iterate and aggregate results until a predefined confidence threshold (e.g., cosine similarity > 0.25) is met, optimizing both precision and compute resources.
* **Selective Encoding & Lazy Searching:** Drastically reduces CPU/Memory overhead. The retrieval engine only embeds and searches modalities explicitly requested by the Agent's tool calls (e.g., bypassing audio index searches if no sound-related context exists).
* **Decoupled Full-Stack Design:** 
  * **Backend:** Built on **FastAPI**, providing asynchronous, high-throughput RESTful endpoints ready for containerization (Docker).
  * **Frontend:** A lightweight **Streamlit** interface for rapid qualitative evaluation, displaying both retrieved frames and the Agent's thought process.
  * **Storage:** Hybrid approach combining **FAISS** for ultra-fast dense vector similarity search and **MongoDB** for flexible metadata enrichment.

## Technology Stack

* **Core AI Models:** `openai/clip-vit-base-patch32` (Visual), `BAAI/bge-m3` (Text/OCR/ASR).
* **Feature Extraction:** PyTorch, YOLOv8 (Objects), PaddleOCR/EasyOCR (Text), Faster-Whisper (Audio).
* **Vector Search & Storage:** FAISS (CPU), MongoDB Atlas.
* **LLM Engine:** Google Gemini 3.6 Flash (`google-genai` SDK).
* **Backend:** FastAPI, Uvicorn, Pydantic.
* **Frontend:** Streamlit.
* **Containerization**: Docker, Docker Compose. 

## High-Level Data Flow

1. `Client` sends a natural language query -> `FastAPI Gateway`.
2. `Retrieval Agent` initiates the ReAct loop to analyze the query context and semantics.
3. **[Agent Loop]** The Agent dispatches specific tool calls (e.g., `search_visual_modality`).
4. **[Agent Loop]** The backend executes FAISS KNN similarity searches on the relevant vector indices.
5. **[Agent Loop]** The Agent observes the returned scores. If the best score meets the quality threshold, the loop terminates; otherwise, it iterates with refined strategies.
6. `MongoDB` enriches the finalized Top-K frames with physical paths and extracted metadata (BLIP captions, YOLO objects, OCR).
7. `Response` is delivered to the Streamlit UI, rendering the Agent's thought process alongside the visually retrieved keyframes. 
   
## Project Structure

```text
AIC_2026/
├── api/
│   ├── main.py                  # FastAPI server and endpoints
│   ├── requirements.txt         # Backend dependencies (google-genai, fastapi, etc.)
│   └── Dockerfile               # Docker configuration for FastAPI Backend
├── ui/ 
│   ├── web_app.py               # Streamlit web interface with Agent Status UX
│   ├── requirements.txt         # Frontend dependencies (streamlit, requests)
│   └── Dockerfile               # Docker configuration for Streamlit Frontend
├── data/                        # Local data (Ignored by Git)
│   ├── keyframes/               # Physical image files (L01, L02, etc.)
│   └── vector_db/               # FAISS index files (*.index)
├── scripts/
│   ├── evaluate.py              # Script to calculate mAP and Recall@K
│   ├── generate_predictions.py  # Script to generate CSV submission for AIC 2026
│   ├── run_search.py            # CLI search execution
│   └── run_vector_indexing.py   # Vectorization pipeline
├── src/
│   ├── agent/
│   │   └── retrieval_agent.py   # The ReAct Agent loop (Brain - uses google-genai)
│   ├── database/
│   │   └── mongo_manager.py     # MongoDB connection and queries
│   ├── retrieval/
│   │   ├── search_engine.py     # Core search logic & aggregation
│   │   └── tools.py             # Atomic tool functions (Visual, Text, Audio)
│   └── vector_search/
│       ├── embedding_models.py  # CLIP and BGE model wrappers
│       └── faiss_manager.py     # FAISS indexing and search
├── docker-compose.yml           # Multi-container orchestration config
├── .env                         # Environment variables (Ignored by Git)
├── .gitignore                   # Git ignore configurations
└── README.md                    # Project documentation 
```

## Installation & Setup (Docker Method - Recommended)
Because the system has been fully containerized, running it via Docker is the easiest and most consistent method. You do not need to install Python or set up virtual environments manually. 

### 1. Prerequisites

Ensure you have Docker and Docker Compose installed on your system. 

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/AIC-2026-Multimodal-Search.git](https://github.com/your-username/AIC-2026-Multimodal-Search.git)
cd AIC-2026-Multimodal-Search
``` 

### 3. Environment Variables 
Create a .env file in the root directory and add your credentials: 
```bash
MONGO_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
GEMINI_API_KEY="your_google_gemini_api_key_here" 
```

### 4. Build and Run 
Launch both the Backend and Frontend simultaneously using Docker Compose:  
```bash
docker-compose up --build 
```
The web app will be automatically accessible at http://localhost:8501 and the API docs at http://localhost:8000/docs. 

## Installation & Setup (Manual Method) 
If you prefer to run the system without Docker:
1. Ensure Python 3.10+ is installed.
2. Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies for both services: 
   ```bash
   pip install -r api/requirements.txt
   pip install -r ui/requirements.txt
   ``` 
4. Set up the .env file as described above.
5. Open Terminal 1 for Backend: ```uvicorn api.main:app --reload --port 8000```
6. Open Terminal 2 for Frontend: ```streamlit run ui/web_app.py``` 

## Future Roadmap: Evaluation & Fine-Tuning
This project is actively under development. While the Agentic ReAct loop and retrieval engines are operational, the immediate next steps involve rigorous quantitative testing and model optimization. 

### Phase 1: Automated Evaluation Pipeline
I am currently developing a robust evaluation script (scripts/evaluate.py) designed to measure system performance against standardized ground truth datasets provided by the AIC organizers. This pipeline will automatically calculate standard Information Retrieval metrics: 

- Recall@K: To ensure relevant keyframes are not missed in the top results.
- Mean Average Precision (mAP): To assess the overall quality and ranking order.
- Mean Reciprocal Rank (MRR): To measure how quickly the first correct frame is returned.

The web app will automatically open in your default browser at http://localhost:8501.

### Phase 2: System Fine-Tuning Strategy
Based on the insights gathered from the Phase 1 evaluation metrics, we plan to implement a multi-tiered fine-tuning strategy to elevate performance:
   1. Level 1 (Weight Tuning): Utilizing automated search algorithms (e.g., Grid Search) to optimize the fusion weights beyond the LLM's dynamic assignments, finding the optimal balance between visual, text, and audio scores.

   2. RLevel 2 (Prompt Engineering): Refining the Gemini Query Agent via Few-Shot Prompting to achieve near-perfect intent parsing, especially for complex, multi-layered queries.
   3. Level 3 (Model Adaptation): If necessary, applying low-rank adaptation (LoRA) or contrastive learning techniques to the core CLIP and BGE models to better understand domain-specific (sports) vocabulary and visual contexts.

   




