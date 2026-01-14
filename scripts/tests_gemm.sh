#!/bin/bash
set -e

python anywhere_main.py --verbosity 10 info  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx


# indexing
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu  --scale_strat random --random_scale_range 0.1 100.0  --mode gemm --split_idx 0
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu  --scale_strat random --random_scale_range 0.1 100.0  --mode gemm --split_idx 2
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu  --scale_strat random --random_scale_range 0.1 100.0  --mode all --split_idx 2


python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation relu --fixed_scale 1 -1
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation leakyrelu --fixed_scale 1 -1
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu --fixed_scale 1 -1


python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation relu
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation leakyrelu
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu


python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation relu    --scale_strat random --random_scale_range 0.1 100.0
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation leakyrelu   --scale_strat random --random_scale_range 0.1 100.0
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --activation prelu  --scale_strat random --random_scale_range 0.1 100.0


# ====== GEMM Mode Tests with Scale Permutations ======

# GEMM with fixed scale (1.0, -1.0) - Default scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with fixed scale (0.5, -0.5)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 0.5 -0.5

# GEMM with fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 2.0 -2.0

# GEMM with fixed scale (10.0, -10.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 10.0 -10.0

# GEMM with random scale (0.1, 100.0) - Default random range
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat random --random_scale_range 0.1 100.0

# GEMM with random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat random --random_scale_range 0.5 50.0

# GEMM with random scale (0.01, 1000.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --scale_strat random --random_scale_range 0.01 1000.0

# GEMM with LeakyReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with LeakyReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --scale_strat fixed --fixed_scale 2.0 -2.0

# GEMM with LeakyReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --scale_strat random --random_scale_range 0.1 100.0

# GEMM with PReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with PReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# GEMM with PReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --scale_strat random --random_scale_range 0.1 100.0

# GEMM with different gemm_tau_strat (midpoint) and fixed scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --gemm_tau_strat midpoint --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with gemm_tau_strat midpoint and random scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --gemm_tau_strat midpoint --scale_strat random --random_scale_range 0.1 100.0

# GEMM with different stable_tau_strat (big) and fixed scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with stable_tau_strat small and fixed scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with combined strategies: midpoint tau, big stable tau, and random scales
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --gemm_tau_strat midpoint --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0

# ====== GEMM with stable_tau_strat and -n parameter variations ======

# GEMM with stable_tau_strat (big) and fixed scales with -n 0
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# GEMM with stable_tau_strat (big) and fixed scales with -n 2
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 2

# GEMM with stable_tau_strat (big) and fixed scales with -n 4
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 4

# GEMM with stable_tau_strat small and fixed scales with -n 0
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# GEMM with stable_tau_strat small and fixed scales with -n 2
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 2

# GEMM with stable_tau_strat small and fixed scales with -n 5
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 5

# GEMM with combined strategies: midpoint tau, big stable tau, and random scales with -n 3
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --gemm_tau_strat midpoint --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 3

# GEMM with combined strategies: midpoint tau, big stable tau, and random scales with -n 4
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation relu --gemm_tau_strat midpoint --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 4

# ====== GEMM with LeakyReLU alpha parameter tests ======

# GEMM with LeakyReLU (alpha=0.01) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with LeakyReLU (alpha=0.1) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with LeakyReLU (alpha=0.2) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --leakyrelu_alpha 0.2 --scale_strat fixed --fixed_scale 2.0 -2.0

# GEMM with LeakyReLU (alpha=0.01) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat random --random_scale_range 0.1 100.0

# GEMM with LeakyReLU (alpha=0.1) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat random --random_scale_range 0.5 50.0

# ====== GEMM with PReLU slope_range parameter tests ======

# GEMM with PReLU (slope=0.01-0.25) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with PReLU (slope=0.1-0.5) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat fixed --fixed_scale 1.0 -1.0

# GEMM with PReLU (slope=0.01-0.1) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --prelu_slope_range 0.01 0.1 --scale_strat fixed --fixed_scale 2.0 -2.0

# GEMM with PReLU (slope=0.01-0.25) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat random --random_scale_range 0.1 100.0

# GEMM with PReLU (slope=0.1-0.5) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib --mode gemm --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat random --random_scale_range 0.5 50.0



echo "All tests completed successfully!"

rm splitted.onnx
