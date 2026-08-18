"""
=============================================================================
FRONTEND WEB UI (STREAMLIT)
=============================================================================
Description:
A user-friendly web interface. It sends user queries to the FastAPI backend,
receives the JSON response, and renders the images and metadata nicely.
=============================================================================
"""

import streamlit as st
import requests
from PIL import Image 
import os  

# Backend API configuration (FastAPI endpoint)
API_URL = os.getenv("API_URL", "http://localhost:8000/api/search") 

# Page configuration
st.set_page_config(page_title="AIC 2026 Multimodal Search", page_icon="🏸", layout="wide") 

st.title("AIC 2026 Multimodal Search System")
st.markdown("Combines BGE-M3 (Text), CLIP (Visual), and Gemini 1.5 (Agent) for query understanding and retrieval.")

# Sidebar for settings and parameters
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results (Top K):", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.markdown("**Note:**")
    st.markdown("- FastAPI backend must be running on port 8000.")

# Main query input
query = st.text_input("Enter your search query:", placeholder="e.g. Badminton player smashing, Olymfit logo on court...")

# Search execution trigger
if st.button("Search", type="primary") or query:
    if not query:
        st.warning("Please enter a search query!")
    else:
        with st.spinner("Gemini Agent is analyzing and searching..."):
            try:
                # Send HTTP POST request to Backend API
                response = requests.post(API_URL, json={"query": query, "top_k": top_k})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Display AI Agent thought process (expandable)
                    with st.expander("View LLM Agent Query Decomposition & Dynamic Weights"):
                        st.json(data.get("agent_thought_process", {}))
                        
                    st.success(f"Found {len(data.get('results', []))} results in {data.get('processing_time_sec')} seconds.")
                    
                    # 2. Render retrieval results
                    results = data.get("results", [])
                    for res in results:
                        st.markdown("---")
                        
                        # Two-column layout: Column 1 for Image, Column 2 for Metadata
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            image_path = res.get("image_path")
                            if os.path.exists(image_path):
                                img = Image.open(image_path)
                                st.image(img, use_container_width=True)
                            else:
                                st.error(f"Image file not found at: {image_path}")
                                
                        with col2:
                            st.subheader(f"Top {res['rank']} - Frame: {res['frame_id']}")
                            st.metric(label="Fusion Score", value=f"{res['score']:.3f}")
                            
                            meta = res.get("metadata", {})
                            st.write(f"**BLIP Caption:** {meta.get('caption', 'N/A')}")
                            st.write(f"**OCR Text:** {meta.get('ocr', 'N/A')}")
                            
                            # Format YOLO detected objects
                            yolo = meta.get('yolo_objects', {})
                            yolo_str = ", ".join([f"{k} ({v})" for k, v in yolo.items()]) if yolo else "N/A"
                            st.write(f"**YOLO Objects:** {yolo_str}")
                            
                else:
                    st.error(f"Backend Error: {response.json().get('detail', response.text)}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Backend! Please ensure FastAPI is running via `uvicorn api.main:app --reload` in another terminal.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}") 