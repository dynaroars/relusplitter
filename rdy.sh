#!/bin/bash
export TOOL_NAME=stable_nas
export CONDA_HOME=$HOME/anaconda3

conda activate $TOOL_NAME


export TOOL_ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${TOOL_ROOT}"
export MKL_SERVICE_FORCE_INTEL=1






