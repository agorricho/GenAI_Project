"""
run_pipeline.py — Milestone 2 end-to-end runner
Runs download_papers.py, waits 120 s, then runs embed_upsert.py.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    print("=== Step 1: Downloading and chunking papers ===")
    subprocess.run([sys.executable, str(HERE / "download_papers.py")], check=True)

    print("\n=== Waiting 120 seconds before embedding... ===")
    time.sleep(120)

    print("\n=== Step 2: Embedding and upserting to Qdrant ===")
    subprocess.run([sys.executable, str(HERE / "embed_upsert.py")], check=True)

    print("\n=== Pipeline complete. ===")
