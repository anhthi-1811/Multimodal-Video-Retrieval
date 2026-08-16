"""
=============================================================================
SUPABASE (POSTGRESQL) DATABASE MANAGER
=============================================================================
Description:
Handles all interactions with Supabase PostgreSQL.
Useful for structured relational data if needed alongside MongoDB.
=============================================================================
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any

class SqlManager:
    def __init__(self, database_url: str):
        """
        Initializes the connection to Supabase PostgreSQL database.
        """
        if not database_url:
            raise ValueError("❌ Supabase Database URL is missing!")
            
        try:
            self.conn = psycopg2.connect(database_url)
            # Enable autocommit for simpler insert/update operations
            self.conn.autocommit = True 
            print("Successfully connected to Supabase PostgreSQL.")
        except psycopg2.OperationalError as e:
            print(f"Failed to connect to Supabase: {e}")
            raise

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Executes a Read query (SELECT) and returns a list of dictionaries.
        RealDictCursor ensures columns are accessible by their names (like JSON).
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                # Fetch all results if it's a SELECT query
                if cursor.description is not None:
                    return cursor.fetchall()
                return []
        except Exception as e:
            print(f"Error executing read query: {e}")
            return []

    def execute_action(self, query: str, params: tuple = None) -> bool:
        """
        Executes an Action query (INSERT, UPDATE, DELETE).
        Returns True if successful.
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
            return True
        except Exception as e:
            print(f"Error executing action query: {e}")
            return False

    def close(self):
        """Closes the PostgreSQL connection."""
        if self.conn:
            self.conn.close()
            print("Supabase connection closed.") 