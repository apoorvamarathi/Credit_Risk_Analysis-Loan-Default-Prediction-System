"""
End-to-End Execution Pipeline
Runs data cleaning, model training, and integration test verification.
"""

import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, desc):
    print(f"\n==================================================")
    print(f"STEP: {desc}")
    print(f"Command: {cmd}")
    print(f"==================================================")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"SUCCESS: {desc} completed successfully.")


if __name__ == "__main__":
    print("Initiating Full End-to-End Pipeline...")
    
    # Step 1: Clean Data & Build SQLite Database
    run_command("python src/data_cleaning.py", "Data Cleaning and Feature Engineering")

    # Step 2: Train Machine Learning Models
    run_command("python src/train_models.py", "Model Training, Cross-Validation and Evaluation")

    # Step 3: Run Automated Test Suite
    run_command("python -m pytest tests/test_pipeline.py -v", "Automated Test Suite Verification")

    print("\n==================================================")
    print("ALL PIPELINE STAGES COMPLETED WITH ZERO ERRORS!")
    print("To launch the interactive dashboard, run:")
    print("  streamlit run app.py")
    print("==================================================")
