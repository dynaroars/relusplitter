#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Compute the paths relative to the script's directory
LIBS_DIR=$(realpath "$SCRIPT_DIR/../libs")
ENVS_DIR=$(realpath "$SCRIPT_DIR/../.envs")
TOOL_NAME="ReluSplitter"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/ReluSplitter.yaml

conda env remove --prefix $CONDA_PREFIX -y
conda env create --prefix $CONDA_PREFIX -f $ENV_FILE_PATH -y
$CONDA_PREFIX/bin/pip install -e $LIBS_DIR/auto_LiRPA