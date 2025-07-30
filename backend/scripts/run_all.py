import subprocess
from pathlib import Path

def run_pipeline():
    steps = [
        "python backend/scripts/generate_labels.py",
        "python backend/data/raw/data/preprocess.py",
        "python backend/src/scripts/train.py"
    ]
    
    for step in steps:
        try:
            subprocess.run(step.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in step {step}: {e}")
            break

if __name__ == "__main__":
    run_pipeline()