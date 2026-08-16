"""
=============================================================================
EVALUATION METRICS SCRIPT
=============================================================================
Description:
Computes standard Information Retrieval metrics (mAP, Recall@K) by comparing 
system predictions against a Ground Truth file.
=============================================================================
"""

import os
import csv
from collections import defaultdict

# ===========================================================================
# 1. HELPER FUNCTIONS TO LOAD DATA
# ===========================================================================
def load_ground_truth(filepath: str) -> dict:
    """
    Reads the ground truth CSV.
    Expected format: query_id, frame_id
    Returns a dictionary: { "query_id": set("frame_id_1", "frame_id_2") }
    """
    gt_dict = defaultdict(set)
    if not os.path.exists(filepath):
        print(f"[WARNING] Ground truth file not found: {filepath}")
        return gt_dict

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row['query_id'].strip()
            f_id = row['frame_id'].strip()
            gt_dict[q_id].add(f_id)
            
    return gt_dict

def load_predictions(filepath: str) -> dict:
    """
    Reads the prediction CSV.
    Expected format: query_id, frame_id, rank
    Returns a dictionary: { "query_id": ["frame_id_1", "frame_id_2", ...] } (Ordered by rank)
    """
    pred_dict = defaultdict(list)
    if not os.path.exists(filepath):
        print(f"[WARNING] Prediction file not found: {filepath}")
        return pred_dict

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row['query_id'].strip()
            f_id = row['frame_id'].strip()
            pred_dict[q_id].append(f_id)
            
    return pred_dict

# ===========================================================================
# 2. METRIC CALCULATIONS
# ===========================================================================
def calculate_metrics(gt_dict: dict, pred_dict: dict, k: int = 100):
    """
    Calculates Mean Average Precision (mAP) and Recall@K.
    """
    total_queries = len(gt_dict)
    if total_queries == 0:
        print("No ground truth data available to evaluate.")
        return 0.0, 0.0

    sum_ap = 0.0
    sum_recall = 0.0

    for q_id, relevant_frames in gt_dict.items():
        # Get top K predictions for this query
        predictions = pred_dict.get(q_id, [])[:k]
        
        hits = 0
        sum_precisions = 0.0
        
        # Calculate Average Precision (AP) and Recall for the current query
        for rank, pred_frame in enumerate(predictions, start=1):
            if pred_frame in relevant_frames:
                hits += 1
                sum_precisions += hits / rank
                
        # AP for this query
        ap = sum_precisions / len(relevant_frames) if relevant_frames else 0.0
        sum_ap += ap
        
        # Recall@K for this query
        recall = hits / len(relevant_frames) if relevant_frames else 0.0
        sum_recall += recall

    # Averages across all queries
    mAP = sum_ap / total_queries
    mean_recall = sum_recall / total_queries

    return mAP, mean_recall

# ===========================================================================
# 3. MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print("==================================================")
    print("   EVALUATION METRICS CALCULATOR   ")
    print("==================================================\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define paths to your CSV files
    # Note: You need to create these files or generate them via run_submission.py
    gt_file = os.path.join(project_root, 'data', 'ground_truth.csv')
    pred_file = os.path.join(project_root, 'data', 'my_predictions.csv')
    
    print(f"[SYSTEM] Loading Ground Truth from: {gt_file}")
    gt_data = load_ground_truth(gt_file)
    
    print(f"[SYSTEM] Loading Predictions from: {pred_file}")
    pred_data = load_predictions(pred_file)
    
    if gt_data and pred_data:
        # Evaluate for Top 100
        K = 100
        print(f"\n[SYSTEM] Calculating metrics for Top-{K}...")
        mAP, recall = calculate_metrics(gt_data, pred_data, k=K)
        
        print("\n==================================================")
        print(f" FINAL RESULTS (Evaluated on {len(gt_data)} queries)")
        print("==================================================")
        print(f" > mAP@{K}    : {mAP:.4f}")
        print(f" > Recall@{K} : {recall:.4f}")
        print("==================================================")
    else:
        print("\n[ERROR] Missing data files. Cannot perform evaluation.")
        print("Ensure 'ground_truth.csv' and 'my_predictions.csv' exist in the 'data/' folder.") 