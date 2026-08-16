import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import declarative_base

# 1. Load environment variables from the .env file
load_dotenv() 
SUPABASE_URL = os.getenv("SUPABASE_URL")

# Fix the URL prefix if necessary (SQLAlchemy requires 'postgresql://' instead of 'postgres://')
if SUPABASE_URL and SUPABASE_URL.startswith("postgres://"):
    SUPABASE_URL = SUPABASE_URL.replace("postgres://", "postgresql://", 1)

# 2. Initialize the database connection (engine)
# The echo=True parameter prints all underlying SQL commands being executed to the console
engine = create_engine(SUPABASE_URL, echo=True)
Base = declarative_base()

# 3. Define the database table schemas  
class Video(Base):
    __tablename__ = 'videos'
    video_id = Column(String, primary_key=True) 
    duration = Column(Float)
    file_path = Column(String) # Path to the .mp4 file on your local machine
    asr_text = Column(Text)    # Spoken dialogue extracted from the video 

class Frame(Base):
    __tablename__ = 'frames'
    frame_id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey('videos.video_id'))
    timestamp = Column(Float)  # The specific second in the video where the frame was extracted
    file_path = Column(String) # Path to the .jpg file on your local machine

class TextData(Base):
    __tablename__ = 'text_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_id = Column(String, ForeignKey('frames.frame_id'))
    ocr_text = Column(Text)    # Text extracted from the image
    caption = Column(Text)     # AI-generated image description

# 4. Execute the table creation on Supabase
if __name__ == "__main__":
    print("Connecting to Supabase...")
    try:
        # This command automatically checks if tables exist. If not, it creates them; otherwise, it skips.
        Base.metadata.create_all(engine)
        print("TABLES CREATED SUCCESSFULLY! Check your Supabase dashboard.")
    except Exception as e:
        print("AN ERROR OCCURRED:", e) 