#!/bin/bash

# Get the absolute path to the script's directory
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT=$SCRIPT_DIR/..

LIBS_DIR=$(realpath "$SCRIPT_DIR/../libs")
ENVS_DIR=$(realpath "$SCRIPT_DIR/../.envs")
TOOL_NAME="ReluSplitter"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/ReluSplitter.yaml


git submodule init
git submodule update


conda env remove --prefix $CONDA_PREFIX
conda env create --prefix $CONDA_PREFIX -f $ENV_FILE_PATH

cd $TOOL_ROOT
git submodule update --init  --recursive
$CONDA_PREFIX/bin/pip install -e $LIBS_DIR/auto_LiRPA
sed -i '865s/self.ori_state_dict)/self.ori_state_dict, strict=False)/' $LIBS_DIR/auto_LiRPA/auto_LiRPA/bound_general.py