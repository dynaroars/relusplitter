#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../../libs")
ENVS_DIR=$(realpath "$SCRIPT_DIR/../../.envs")
TOOL_NAME="nnenum"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/nnenum.txt

mkdir -p $LIBS_DIR
cd  $LIBS_DIR



git clone https://github.com/stanleybak/nnenum
cd nnenum
git checkout cf7c0e72c13543011a7ac3fbe0f5c59c3aafa77e

# eval "$(conda shell.bash hook)"
conda env remove --prefix $CONDA_PREFIX -y
conda create --prefix $CONDA_PREFIX python=3.10 -y
# conda activate $CONDA_PREFIX
$CONDA_PREFIX/bin/pip install -r $ENV_FILE_PATH
# pip install -r $ENVS_DIR/nnenum.txt