"""
=============================================================================
FAISS MANAGER (VECTOR DATABASE)
=============================================================================
Description:
Handles the creation, updating, and searching of FAISS vector indices.
Manages the translation between FAISS integer IDs and string IDs (like frame_id)
using a JSON mapping file.
=============================================================================
"""

import os
import json
import numpy as np
import faiss

class FaissManager:
    def __init__(self, index_path: str, map_path: str, dimension: int):
        """
        Initializes the FAISS Index and its corresponding ID map.
        If the files already exist on disk, it loads them into RAM.
        Otherwise, it creates a new empty index.
        """
        self.index_path = index_path
        self.map_path = map_path
        self.dimension = dimension

        # Dictionary to map FAISS integer IDs to String IDs (e.g., { "0": "L21_V001_020" })
        self.id_map = {}

        # Load existing index if it exists, else create a new one
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            print(f"Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            
            with open(self.map_path, 'r', encoding='utf-8') as f:
                self.id_map = json.load(f)
                
            print(f"Successfully loaded {self.index.ntotal} vectors.")
        else:
            print(f"Creating new FAISS index (Dimension: {self.dimension})...")
            # IndexFlatIP is used for Cosine Similarity (requires normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, vector: list, item_id: str):
        """
        Adds a single vector and its string ID to the index.
        """
        if not vector:
            return

        # FAISS requires vectors to be in numpy float32 format
        vec_np = np.array([vector], dtype=np.float32)
        
        # Get the next available integer ID in FAISS
        current_faiss_id = self.index.ntotal
        
        # Add vector to the FAISS index
        self.index.add(vec_np)
        
        # Save the mapping (Convert integer ID to string because JSON keys must be strings)
        self.id_map[str(current_faiss_id)] = item_id

    def add_batch(self, vectors: list, item_ids: list):
        """
        Adds multiple vectors at once (Much faster for large datasets).
        """
        if not vectors or len(vectors) != len(item_ids):
            raise ValueError("Vectors list and item_ids list must have the same length and cannot be empty.")
            
        vec_np = np.array(vectors, dtype=np.float32)
        start_faiss_id = self.index.ntotal
        
        self.index.add(vec_np)
        
        # Map the sequence of new integer IDs to the provided string IDs
        for i, item_id in enumerate(item_ids):
            self.id_map[str(start_faiss_id + i)] = item_id

    def save(self):
        """
        Persists the FAISS index and the JSON map to the local disk.
        """
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS .index file
        faiss.write_index(self.index, self.index_path)
        
        # Save JSON mapping file
        with open(self.map_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=4)
            
        print(f"Saved {self.index.ntotal} vectors to {self.index_path}")

    def search(self, query_vector: list, top_k: int = 5) -> tuple:
        """
        Searches for the Top K most similar vectors.
        Returns a tuple: (list of string_ids, list of similarity_scores)
        """
        if self.index.ntotal == 0 or not query_vector:
            return [], []

        # Convert query vector to numpy float32
        query_np = np.array([query_vector], dtype=np.float32)
        
        # Perform similarity search
        # distances: The similarity scores (higher is better for Inner Product)
        # indices: The FAISS integer IDs
        distances, indices = self.index.search(query_np, top_k)
        
        # Flatten the 2D arrays to 1D lists
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        result_ids = []
        result_scores = []
        
        for dist, idx in zip(distances, indices):
            # FAISS returns -1 if it cannot find enough results
            if idx != -1:
                # Translate FAISS integer ID back to the original String ID
                original_id = self.id_map.get(str(idx), "UNKNOWN_ID")
                
                result_ids.append(original_id)
                result_scores.append(round(dist, 4))
                
        return result_ids, result_scores 