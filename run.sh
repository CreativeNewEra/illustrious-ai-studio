#!/bin/bash
# Illustrious AI Studio Launch Script

echo "Starting Illustrious AI Studio..."

# Activate conda environment and run
$(which conda) run -n illustrious python main.py "$@"
