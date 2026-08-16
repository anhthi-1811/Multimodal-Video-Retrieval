"""
=============================================================================
MONGODB DATABASE MANAGER
=============================================================================
Description: 
Handles all interactions with MongoDB Atlas.
Provides methods to upsert frame data and retrieve metadata.
=============================================================================
"""

from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Any 

class MongoManager:
    def __init__(self, uri: str, db_name: str = 'aic_2026_db', collection_name: str = 'keyframes_data'):
        """
        Initializes the MongoDB connection. 
        """
        if not uri: 
            raise ValueError("MongoDB URI is missing!")
            
        try:
            self.client = MongoClient(uri)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.collection.create_index("frame_id", unique=True)  
            print(f"Successfully connected to MongoDB Atlas (DB: {db_name})")
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB Atlas: {e}")
            raise

    def upsert_frame_data(self, document_id: str, payload: Dict[str, Any], id_field: str = "frame_id") -> bool:
        """
        Updates an existing document or creates a new one if it doesn't exist.
        Used extensively in OCR, YOLO, BLIP pipelines (default: frame_id) 
        and Audio pipelines (custom: video_id).
        """
        try:
            self.collection.update_one(
                {id_field: document_id},
                {"$set": payload},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error upserting data for {id_field} {document_id}: {e}")
            return False 

    def upsert_frames_batch(self, payloads: List[Dict[str, Any]]) -> bool:
        """
        Batch upsert multiple frame documents into MongoDB.

        Instead of sending one request per frame, this method
        sends many upsert operations in a single bulk_write request.

        This is significantly faster for large-scale ingestion.
        """

        if not payloads:
            return True

        try:
            operations = []

            for payload in payloads:
                frame_id = payload.get("frame_id")

                if not frame_id:
                    continue

                operations.append(
                    UpdateOne(
                        {"frame_id": frame_id},
                        {"$set": payload},
                        upsert=True
                    )
                )

            if not operations:
                return True

            result = self.collection.bulk_write(
                operations,
                ordered=False 
            )

            return True

        except Exception as e:
            print(
                f"Error during batch upsert "
                f"({len(payloads)} frames): {e}"
            )
            return False

    def get_metadata_by_ids(self, frame_ids: List[str]) -> Dict[str, dict]:
        """
        Retrieves full documents for a list of frame_ids.
        Returns a dictionary for O(1) fast lookup during Reranking. 
        Format: {"L21_V001_020": {document_data}, ...}
        """
        try:
            docs = self.collection.find({"frame_id": {"$in": frame_ids}})
            # Convert to dictionary with frame_id as key 
            return {doc['frame_id']: doc for doc in docs} 
        except Exception as e:
            print(f"Error retrieving metadata: {e}")
            return {}

    def close(self):
        """Closes the MongoDB connection."""
        self.client.close()
        print("MongoDB connection closed.") 