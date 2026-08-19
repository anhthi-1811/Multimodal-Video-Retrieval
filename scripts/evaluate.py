"""
=============================================================================
EVALUATION METRICS SCRIPT (mAP, Recall@K, MRR)
=============================================================================
Description:
Computes standard Information Retrieval metrics (mAP, Recall@K, MRR) by 
comparing system predictions against a Ground Truth file.
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
    Returns: { "query_id": set("frame_id_1", "frame_id_2") }
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
    Reads the prediction CSV and sorts frames explicitly by rank.
    Expected format: query_id, frame_id, rank
    Returns: { "query_id": ["frame_id_1", "frame_id_2", ...] }
    """
    raw_preds = defaultdict(list)
    if not os.path.exists(filepath):
        print(f"[WARNING] Prediction file not found: {filepath}")
        return {}

    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row['query_id'].strip()
            f_id = row['frame_id'].strip()
            rank = int(row.get('rank', 0))
            raw_preds[q_id].append((rank, f_id))

    # Sort each query's predictions by rank to guarantee correct order
    pred_dict = {}
    for q_id, items in raw_preds.items():
        items.sort(key=lambda x: x[0])
        pred_dict[q_id] = [f_id for _, f_id in items]

    return pred_dict

# ===========================================================================
# 2. METRIC CALCULATIONS
# ===========================================================================
def calculate_metrics(gt_dict: dict, pred_dict: dict, k: int = 100):
    """
    Calculates Mean Average Precision (mAP@K), Recall@K, and Mean Reciprocal Rank (MRR@K).
    """
    total_queries = len(gt_dict)
    if total_queries == 0:
        print("No ground truth data available to evaluate.")
        return 0.0, 0.0, 0.0

    sum_ap = 0.0
    sum_recall = 0.0
    sum_rr = 0.0

    for q_id, relevant_frames in gt_dict.items():
        raw_predictions = pred_dict.get(q_id, [])[:k]
        
        # Deduplicate predictions while preserving order
        predictions = []
        seen = set()
        for f_id in raw_predictions:
            if f_id not in seen:
                seen.add(f_id)
                predictions.append(f_id)

        hits = 0
        sum_precisions = 0.0
        first_hit_rank = None

        for rank, pred_frame in enumerate(predictions, start=1):
            if pred_frame in relevant_frames:
                hits += 1
                sum_precisions += hits / rank
                if first_hit_rank is None:
                    first_hit_rank = rank

        # 1. Average Precision (AP)
        ap = sum_precisions / len(relevant_frames) if relevant_frames else 0.0
        sum_ap += ap

        # 2. Recall@K
        recall = hits / len(relevant_frames) if relevant_frames else 0.0
        sum_recall += recall

        # 3. Reciprocal Rank (RR)
        rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        sum_rr += rr

    mAP = sum_ap / total_queries
    mean_recall = sum_recall / total_queries
    mrr = sum_rr / total_queries

    return mAP, mean_recall, mrr

# ===========================================================================
# 3. MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print("==================================================")
    print("       EVALUATION METRICS CALCULATOR              ")
    print("==================================================\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    gt_file = os.path.join(project_root, 'data', 'ground_truth.csv')
    pred_file = os.path.join(project_root, 'data', 'my_predictions.csv')
    
    print(f"[SYSTEM] Loading Ground Truth from: {gt_file}")
    gt_data = load_ground_truth(gt_file)
    
    print(f"[SYSTEM] Loading Predictions from: {pred_file}")
    pred_data = load_predictions(pred_file)
    
    if gt_data and pred_data:
        K = 100
        print(f"\n[SYSTEM] Calculating metrics for Top-{K}...")
        mAP, recall, mrr = calculate_metrics(gt_data, pred_data, k=K)
        
        print("\n==================================================")
        print(f" FINAL RESULTS (Evaluated on {len(gt_data)} queries)")
        print("==================================================")
        print(f" > mAP@{K}     : {mAP:.4f}")
        print(f" > Recall@{K}  : {recall:.4f}")
        print(f" > MRR@{K}     : {mrr:.4f}")
        print("==================================================")
    else:
        print("\n[ERROR] Missing data files. Cannot perform evaluation.")
        print("Ensure 'ground_truth.csv' and 'my_predictions.csv' exist in the 'data/' folder.") 