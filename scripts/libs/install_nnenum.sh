#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
mkdir -p $LIBS_DIR
cd  $LIBS_DIR



git clone https://github.com/stanleybak/nnenum
cd nnenum
git checkout cf7c0e72c13543011a7ac3fbe0f5c59c3aafa77e


conda deactivate; conda env remove --name nnenum
conda env create -n nnenum python=3.10
conda activate nnenum
pip install -r requirements.txt
