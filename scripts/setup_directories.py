"""
Setup Directory Structure

Creates the necessary directories for the AI test generator project.
"""

import os
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

# Directories to create
directories = [
    "training_data/automation_exercise",
    "logs",
    "screenshots/automation_exercise",
    "reports/automation_exercise"
]

def main():
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    main()
