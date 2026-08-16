"""
=============================================================================
FRAME EXTRACTION PIPELINE
=============================================================================
Description:
Scans the 'raw_videos' directory, finding all video files.
Uses FFmpeg to extract frames (default: 1 frame per second) sequentially
from L21 to L30 and saves them into the structured 'keyframes' directory.
=============================================================================
"""

import os
import subprocess
import time

# ---------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, 'data')

RAW_VIDEOS_DIR = os.path.join(DATA_ROOT, 'raw_videos')
KEYFRAMES_OUT_DIR = os.path.join(DATA_ROOT, 'keyframes')


class FfmpegExtractor:
    def __init__(self, fps: int = 1):
        """
        Initializes the frame extractor.
        Default rate: 1 frame per second (fps=1).
        """
        self.fps = fps
        # Verify that FFmpeg is installed and accessible in the system PATH
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except Exception:
            raise EnvironmentError("FFmpeg is not installed or not found in system PATH.")

    def process(self, video_path: str, output_folder: str) -> bool:
        """
        Executes FFmpeg to extract frames from a video file.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Output filename pattern: 001.jpg, 002.jpg, ...
        output_pattern = os.path.join(output_folder, "%03d.jpg")
        
        # FFmpeg command for fixed-FPS extraction
        command = [
            'ffmpeg',
            '-y',                     # Overwrite output files without asking
            '-i', video_path,         # Input video file path
            '-vf', f'fps={self.fps}',   # Video filter: Frames per second
            '-q:v', '2',              # Image quality scale (2 is high quality)
            output_pattern
        ]
        
        try:
            # Run command in background and suppress verbose FFmpeg console logs
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] FFmpeg failed on {os.path.basename(video_path)}: {e}")
            return False


def main():
    print("==================================================")
    print("STARTING FRAME EXTRACTION PIPELINE")
    print("==================================================\n")

    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"Error: Directory not found -> {RAW_VIDEOS_DIR}")
        return

    extractor = FfmpegExtractor(fps=1)
    
    # -----------------------------------------------------------------------
    # 2. DISCOVER & SORT VIDEO PATHS (Ensures sequential order L21 -> L30)
    # -----------------------------------------------------------------------
    video_paths = []
    for dirpath, _, filenames in os.walk(RAW_VIDEOS_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_paths.append(os.path.join(dirpath, filename))

    # Sort paths alphabetically to guarantee L21_a -> L22_a -> ... -> L30_a order
    video_paths = sorted(video_paths)

    total_videos = len(video_paths)
    print(f"Discovered {total_videos} videos to extract sequentially.\n")

    start_time = time.time()

    # -----------------------------------------------------------------------
    # 3. EXTRACTION LOOP
    # -----------------------------------------------------------------------
    for index, video_path in enumerate(video_paths, start=1):
        # Reconstruct output folder structure
        # Example Input: .../raw_videos/Videos_L21_a/L21_V001/L21_V001.mp4
        filename = os.path.basename(video_path)
        video_id = os.path.splitext(filename)[0]  # "L21_V001"
        
        # Get relative path from RAW_VIDEOS_DIR to resolve the parent batch directory
        rel_path = os.path.relpath(video_path, RAW_VIDEOS_DIR)
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 2:
            batch_folder = parts[0]  # "Videos_L21_a"
            
            # Map "Videos_L21_a" -> "Keyframes_L21" to maintain compatibility with ingestion pipeline
            batch_parts = batch_folder.split('_')
            if len(batch_parts) >= 2:
                out_batch_folder = f"Keyframes_{batch_parts[1]}"  # "Keyframes_L21"
            else:
                out_batch_folder = batch_folder
        else:
            out_batch_folder = "Keyframes_Misc"

        # Construct destination directory for extracted frames
        output_folder = os.path.join(KEYFRAMES_OUT_DIR, out_batch_folder, "keyframes", video_id)
        
        print(f"[{index}/{total_videos}] Extracting: {video_id} -> {out_batch_folder}/...")
        
        # Resume mechanism: Skip processing if output directory already contains frames
        if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
            print("  -> Skipping. Frames already exist.")
            continue
            
        success = extractor.process(video_path, output_folder)
        if success:
            extracted_count = len(os.listdir(output_folder))
            print(f"  -> Done. Extracted {extracted_count} frames.")

    elapsed_time = time.time() - start_time
    print(f"\nEXTRACTION COMPLETED in {elapsed_time:.2f} seconds!")
    print("==================================================")


if __name__ == "__main__":
    main()  