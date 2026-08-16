"""
=============================================================================
TEST INDEX GENERATOR (MINI-INDEX FOR EVALUATION)
=============================================================================
Description:
This script creates a lightweight, isolated vector database exclusively for 
evaluation purposes. It guarantees the inclusion of predefined "Ground Truth" 
images and mixes them with randomly sampled "Noise" images to simulate a 
realistic but much faster retrieval scenario.
=============================================================================
"""

import os
import sys
import json
import time
import random
import numpy as np

# ---------------------------------------------------------------------------
# 1. PATH CONFIGURATION & MODULE IMPORT
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.vector_search.clip_embedder import ClipEmbedder

# Define physical paths
KEYFRAMES_DIR = os.path.join(project_root, 'data', 'keyframes')
VECTOR_DB_DIR = os.path.join(project_root, 'data', 'vector_database')

# Output files specifically named for testing to avoid overwriting production data
TEST_MATRIX_PATH = os.path.join(VECTOR_DB_DIR, 'test_features.npy')
TEST_MAPPING_PATH = os.path.join(VECTOR_DB_DIR, 'test_mapping.json')

# ---------------------------------------------------------------------------
# 2. TEST CONFIGURATION
# ---------------------------------------------------------------------------
# The exact frame IDs we are testing in our notebook (Ground Truths)
TARGET_IDS = [
    "L21_V001_150",
    "L21_V002_085",
    "L21_V003_210",
    "L21_V001_405",
    "L21_V005_022"
]

# How many random images to add to confuse the model (Noise)
NUM_NOISE_IMAGES = 495 

def main():
    print("==================================================")
    print("STARTING TEST INDEX GENERATOR")
    print("==================================================\n")

    # 1. Initialize Embedder
    print("[1/4] Initializing CLIP Embedder...")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    clip = ClipEmbedder()
    
    # 2. Scan Directory & Segregate Data
    print("[2/4] Scanning Keyframes & Segregating Targets vs Noise...")
    target_paths = []
    noise_pool = []
    
    for dirpath, _, filenames in os.walk(KEYFRAMES_DIR):
        for filename in filenames:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(dirpath, filename)
            folder_name = os.path.basename(dirpath)
            frame_num = os.path.splitext(filename)[0]
            frame_id = f"{folder_name}_{frame_num}"
            
            if frame_id in TARGET_IDS:
                target_paths.append((frame_id, img_path))
            else:
                noise_pool.append((frame_id, img_path))
                
    print(f"Found {len(target_paths)}/{len(TARGET_IDS)} target images.")
    
    # Randomly select noise images
    random.seed(42) # Set seed for reproducibility
    selected_noise = random.sample(noise_pool, min(NUM_NOISE_IMAGES, len(noise_pool)))
    print(f"Selected {len(selected_noise)} random noise images.")

    # Combine and shuffle the test dataset
    test_dataset = target_paths + selected_noise
    random.shuffle(test_dataset)
    total_test_images = len(test_dataset)
    
    # 3. Embedding Loop
    print("\n[3/4] Extracting Vectors for Mini-Index...")
    all_features = []
    frame_id_mapping = []
    
    start_time = time.time()
    for index, (frame_id, img_path) in enumerate(test_dataset, start=1):
        try:
            vector = clip.embed_image(img_path)
            if vector is not None:
                all_features.append(vector)
                frame_id_mapping.append(frame_id)
                
            if index % 100 == 0 or index == total_test_images:
                print(f"  -> Embedded {index}/{total_test_images} frames")
        except Exception as e:
            print(f"Error at {frame_id}: {e}")

    # 4. Compile & Export
    print("\n[4/4] Compiling Test Matrix and Exporting...")
    feature_matrix = np.vstack(all_features)
    
    np.save(TEST_MATRIX_PATH, feature_matrix)
    with open(TEST_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(frame_id_mapping, f, ensure_ascii=False, indent=4)
        
    print(f"\nTEST INDEX GENERATED in {time.time() - start_time:.2f} seconds!")
    print(f"Test Matrix Shape: {feature_matrix.shape}")
    print(f"Matrix saved to: {TEST_MATRIX_PATH}")
    print(f"Mapping saved to: {TEST_MAPPING_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main() 