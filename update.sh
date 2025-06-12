#!/bin/bash
# Update Illustrious AI Studio

echo "Updating Illustrious AI Studio..."

# Pull latest changes
git pull

# Run setup with force flag to update dependencies
python setup.py --force --skip-models

echo "Update complete!"
