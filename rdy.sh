#!/bin/bash
export TOOL_NAME=ReluSplitter

conda activate .envs/$TOOL_NAME


export TOOL_ROOT=$(pwd)
export PYTHONPATH="${PYTHONPATH}:${TOOL_ROOT}:${TOOL_ROOT}/libs"
export MKL_SERVICE_FORCE_INTEL=1


# disable e-cores for consistent performance
for i in {16..23}; do echo 0 | sudo tee /sys/devices/system/cpu/cpu${i}/online; done


