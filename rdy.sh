#!/bin/bash
export TOOL_NAME=ReluSplitter

conda activate .envs/$TOOL_NAME


export TOOL_ROOT=$(pwd)
export LIB_PATH=$(pwd)/libs
export PYTHONPATH="${PYTHONPATH}:${TOOL_ROOT}:${TOOL_ROOT}/libs"
export MKL_SERVICE_FORCE_INTEL=1
export GRB_LICENSE_FILE="${LIB_PATH}/gurobi.lic" 


# use one of them to fix the closeness check for Conv
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export CUBLAS_WORKSPACE_CONFIG=:16:8



# disable e-cores for consistent performance    12700k
# for i in {16..23}; do echo 0 | sudo tee /sys/devices/system/cpu/cpu${i}/online; done


