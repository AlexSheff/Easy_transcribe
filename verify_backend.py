import sys
import os
import logging
from app.pipeline.transcription_pipeline import TranscriptionPipeline

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pipeline():
    # The file path from the directory listing
    test_file = "2026-04-24 17-39-48 Alex Den.mp4"
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file '{test_file}' not found!")
        return

    print(f"Starting test with file: {test_file}")
    
    try:
        pipeline = TranscriptionPipeline()
        print("Pipeline initialized successfully.")
        
        print("Processing file (this may take a while)...")
        results, export_path = pipeline.process_file(test_file)
        
        print("\nSUCCESS!")
        print(f"Exported to: {export_path}")
        print(f"Number of segments: {len(results)}")
        
        # Check if export file exists
        if os.path.exists(export_path):
            print(f"Verified export file exists at: {export_path}")
        else:
            print(f"ERROR: Export file NOT found at: {export_path}")
            
    except Exception as e:
        print(f"\nFAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
