"""
=============================================================================
DATABASE PACKAGE
=============================================================================
Description:
Provides manager classes to handle connections and CRUD operations 
with external databases (MongoDB).
=============================================================================
"""

# Import the class directly into the package namespace
from .mongo_manager import MongoManager 

# __all__ restricts what gets imported when someone uses `from src.database import *`
__all__ = [
    'MongoManager'
] 