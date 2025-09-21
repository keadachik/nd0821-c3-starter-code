#!/bin/bash
# Build script for Render deployment
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training model..."
cd starter
python train_model.py
cd ..

echo "Build complete!"

