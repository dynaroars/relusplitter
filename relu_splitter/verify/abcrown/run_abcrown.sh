#!/bin/bash

# export MKL_SERVICE_FORCE_INTEL=1
cmd="$CONDA_HOME/envs/abcrown_24/bin/python $TOOL_ROOT/libs/alpha-beta-CROWN/complete_verifier/abcrown.py --config $TOOL_ROOT/src/verify/abcrown/onnx_with_one_vnnlib.yaml  $@"
echo $cmd
$cmd
