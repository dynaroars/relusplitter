#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
mkdir -p $LIBS_DIR
cd  $LIBS_DIR


git clone https://github.com/dynaroars/neuralsat
cd neuralsat
git checkout 629cc6b02472c46ed4792f0ad6722b947081ebdc

conda deactivate; conda env remove --name neuralsat
conda env create -f env.yaml
# pip install "third_party/haioc"


