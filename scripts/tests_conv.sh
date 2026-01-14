#!/bin/bash
set -e

# ====== CONV Mode Tests - cifar_base_kw instances ======
python anywhere_main.py --verbosity 10 info  --net stuff/cifar_base_kw.onnx  

# indexing
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 0
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 1
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode all --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 1

# CONV with fixed scale (1.0, -1.0) - Default scales
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with fixed scale (0.5, -0.5)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 0.5 -0.5

# CONV with fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with fixed scale (10.0, -10.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 10.0 -10.0

# CONV with random scale (0.1, 100.0) - Default random range
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# CONV with random scale (0.01, 1000.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.01 1000.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with LeakyReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --scale_strat random --random_scale_range 0.1 100.0

# CONV with PReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with PReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --scale_strat random --random_scale_range 0.1 100.0

# ====== CONV Mode Tests - cifar_wide_kw instances ======

# CONV with fixed scale (1.0, -1.0) on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with random scale (0.1, 100.0) on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0) on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0) on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with random scale (0.5, 50.0) on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV Mode Tests - cifar_deep_kw instances ======

# CONV with fixed scale (1.0, -1.0) on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with random scale (0.1, 100.0) on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0) on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0) on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with random scale (0.5, 50.0) on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV with stable_tau_strat variations ======

# CONV with stable_tau_strat (big) on base network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with stable_tau_strat (small) on base network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with stable_tau_strat (big) and random scales on wide network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0

# CONV with stable_tau_strat (small) and random scales on deep network
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0

# ====== CONV with stable_tau_strat and -n parameter variations ======

# CONV with stable_tau_strat (big) on base network with -n 0
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# CONV with stable_tau_strat (big) on base network with -n 2
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 2

# CONV with stable_tau_strat (big) on base network with -n 5
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 5

# CONV with stable_tau_strat (small) on base network with -n 0
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# CONV with stable_tau_strat (small) on base network with -n 3
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 3

# CONV with stable_tau_strat (big) and random scales on wide network with -n 2
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 2

# CONV with stable_tau_strat (big) and random scales on wide network with -n 4
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 4

# CONV with stable_tau_strat (small) and random scales on deep network with -n 0
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0 -n 0

# CONV with stable_tau_strat (small) and random scales on deep network with -n 4
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0 -n 4

# ====== CONV with LeakyReLU alpha parameter tests ======

# CONV with LeakyReLU (alpha=0.01) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU (alpha=0.1) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU (alpha=0.2) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.2 --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with LeakyReLU (alpha=0.01) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU (alpha=0.1) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV with PReLU slope_range parameter tests ======

# CONV with PReLU (slope=0.01-0.25) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU (slope=0.1-0.5) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU (slope=0.01-0.1) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.1 --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with PReLU (slope=0.01-0.25) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat random --random_scale_range 0.1 100.0

# CONV with PReLU (slope=0.1-0.5) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat random --random_scale_range 0.5 50.0


echo "All tests completed successfully!"

rm splitted.onnx
