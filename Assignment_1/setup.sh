#!/usr/bin/bash

# Create virtual environment
python -m venv env

# Activate environment
source ./env/bin/activate

# System dependencies for OpenCV
sudo apt-get update
sudo apt-get install -y python3-opencv

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Deactivate the environment
deactivate
