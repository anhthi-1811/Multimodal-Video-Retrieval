"""
=============================================================================
BATCH INFERENCE (PREDICTION GENERATOR)
=============================================================================
Description:
Reads a list of test queries provided by the AIC organizers, feeds each 
through the AI Retrieval Agent, and exports the retrieved top-K frame IDs 
into a CSV format suitable for the evaluation pipeline.
=============================================================================
"""

import os
import sys
import csv
import time

# Set relative paths to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.agent.retrieval_agent import RetrievalAgent

def generate_predictions(input_csv: str, output_csv: str, top_k: int = 50):
    """
    Reads queries from an input CSV file, runs the RetrievalAgent ReAct loop 
    for each query, and saves the predicted top-K frame IDs to an output CSV file.
    """
    print(f"Loading queries from: {input_csv}")
    print(f"Results will be saved to: {output_csv}\n")
    
    # 1. Initialize the Agent
    agent = RetrievalAgent()
    
    # 2. Load the input query dataset
    try:
        with open(input_csv, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            queries = list(reader)
    except FileNotFoundError:
        print(f"[ERROR] Could not find file {input_csv}. Please verify the path.")
        return

    # Open output file for streaming predictions
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        # Standard format: query_id and comma-separated predicted frame IDs
        fieldnames = ['query_id', 'predicted_frames']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_queries = len(queries)
        
        # 3. Iterate through each query in the dataset
        for idx, row in enumerate(queries, 1):
            query_id = row.get('query_id', f"Q_{idx:03d}")
            # Organize query text based on column headers provided in dataset
            query_text = row.get('query', '') 
            
            if not query_text:
                print(f"[{idx}/{total_queries}] Skipping empty query text.")
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing Query [{idx}/{total_queries}]: {query_id}")
            print(f"Text: '{query_text}'")
            print(f"{'='*50}")
            
            # Execute search through the Agent
            try:
                results = agent.run(user_query=query_text, score_threshold=0.25)
                
                # Extract Top-K frame IDs from the scored tuples
                top_frames = [frame_id for frame_id, score in results[:top_k]]
                
                # Write to CSV formatted as comma-separated frame IDs
                writer.writerow({
                    'query_id': query_id,
                    'predicted_frames': ",".join(top_frames)
                })
                
                # Sleep briefly to comply with API rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR processing query {query_id}]: {e}")
                
    print(f"\n[SUCCESS] Saved predictions for {total_queries} queries to: {output_csv}")

if __name__ == "__main__":
    # Configure input and output paths relative to project root
    INPUT_FILE = os.path.join(project_root, "data", "queries.csv")
    OUTPUT_FILE = os.path.join(project_root, "data", "my_predictions.csv")
    
    generate_predictions(input_csv=INPUT_FILE, output_csv=OUTPUT_FILE, top_k=50) 