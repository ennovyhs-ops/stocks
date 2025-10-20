#!/bin/bash

# Set working directory to the folder where this script lives
cd "$(dirname "$0")"

# Activate Python environment if needed (uncomment if using virtualenv)
# source venv/bin/activate

# Run the analysis script
/usr/bin/python3 auto_analysis.py

# Optional: log output
echo "Run completed at $(date)" >> analysis_log.txt
