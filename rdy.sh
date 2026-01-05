#!/bin/bash
export TOOL_NAME=ReluSplitter

conda activate .envs/$TOOL_NAME


export TOOL_ROOT=$(pwd)
export LIB_PATH=$(pwd)/libs
export PYTHONPATH="${PYTHONPATH}:${TOOL_ROOT}:${TOOL_ROOT}/libs"
export MKL_SERVICE_FORCE_INTEL=1
export GRB_LICENSE_FILE="${LIB_PATH}/gurobi.lic" 
