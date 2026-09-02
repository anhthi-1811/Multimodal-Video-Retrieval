"""
=============================================================================
FRONTEND WEB UI (STREAMLIT) - AGENTIC VERSION
=============================================================================
Description:
A user-friendly web interface. It sends user queries to the FastAPI backend,
displays the AI Agent's thought process, and renders retrieved images and metadata.
=============================================================================
"""

import streamlit as st
import requests 
from PIL import Image 
import os  

# Backend API configuration (FastAPI endpoint)
# Uses environment variable if deployed, falls back to localhost for local testing
API_URL = os.getenv("API_URL", "http://localhost:8000/api/search") 

# Page configuration for a wide layout and appropriate branding 
st.set_page_config(page_title="AIC 2026 AI Agent Search", page_icon="🤖", layout="wide") 

st.title("AIC 2026 Multimodal AI Agent")
st.markdown("Powered by **Gemini 1.5 Tool Calling**, this agent autonomously routes queries to Visual, OCR, or ASR tools, evaluates results, and iterates to find the best keyframes.")

# Sidebar for settings and architecture notes
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results (Top K):", min_value=1, max_value=50, value=15)
    st.markdown("---")
    st.markdown("**Architecture Update:**")
    st.markdown("- Static JSON routing has been replaced by an autonomous ReAct loop.")
    st.markdown("- Backend FastAPI must be running and accessible.")

# Main search bar
query = st.text_input("Enter your search query:", placeholder="e.g. Badminton player smashing, Yonex logo on court...")

# Execute search when button is clicked or Enter is pressed
if st.button("Search", type="primary") or query:
    if not query:
        st.warning("Please enter a search query!")
    else:
        # Use st.status to simulate the Agent's reasoning steps for better UX
        with st.status("Agent is analyzing and searching...", expanded=True) as status:
            st.write("1. Understanding user intent...")
            st.write("2. Invoking appropriate tools (Visual/OCR/ASR)...")
            st.write("3. Evaluating similarity scores and iterating...")
            
            try:
                # Dispatch HTTP POST request to the Agentic Backend
                response = requests.post(API_URL, json={"query": query, "top_k": top_k})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update status box upon successful retrieval
                    status.update(
                        label=f"{data.get('agent_status', 'ReAct Loop Complete')} in {data.get('processing_time_sec')}s", 
                        state="complete", 
                        expanded=False
                    )
                    
                    st.success(f"Found top {len(data.get('results', []))} keyframes.")
                    
                    # Iteratively render the retrieval results
                    results = data.get("results", [])
                    for res in results:
                        st.markdown("---")
                        
                        # Set up a two-column layout: Column 1 (Image), Column 2 (Metadata)
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            image_path = res.get("image_path")
                            # Verify if the physical image exists before rendering
                            if image_path and os.path.exists(image_path):
                                img = Image.open(image_path)
                                st.image(img, use_container_width=True)
                            else:
                                st.error(f"Image file not found locally: {image_path}")
                                
                        with col2:
                            # Display rank, Frame ID, and retrieval score
                            st.subheader(f"Top {res['rank']} - Frame: {res['frame_id']}")
                            st.metric(label="Retrieval Score", value=f"{res['score']:.4f}")
                            
                            # Safely extract metadata fields
                            meta = res.get("metadata", {})
                            st.write(f"**BLIP Caption:** {meta.get('caption', 'N/A')}")
                            st.write(f"**OCR Text:** {meta.get('ocr', 'N/A')}")
                            
                            # Safely format YOLO objects dictionary into a string
                            yolo = meta.get('yolo_objects', {})
                            if isinstance(yolo, dict) and yolo:
                                yolo_str = ", ".join([f"{k} ({v})" for k, v in yolo.items()])
                            else:
                                yolo_str = "N/A"
                            st.write(f"**YOLO Objects:** {yolo_str}")
                            
                else:
                    # Handle errors returned by the FastAPI backend
                    status.update(label="API Error", state="error", expanded=True)
                    st.error(f"Backend Error: {response.json().get('detail', response.text)}")
                    
            except requests.exceptions.ConnectionError:
                # Handle cases where the Backend server is down or unreachable
                status.update(label="Connection Failed", state="error", expanded=True)
                st.error("Cannot connect to Backend! Please ensure FastAPI is running (check docker-compose or uvicorn).")
            except Exception as e:
                # Catch-all for unexpected frontend crashes
                status.update(label="Unexpected Error", state="error", expanded=True)
                st.error(f"An unexpected error occurred: {e}")