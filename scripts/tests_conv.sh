#!/bin/bash
set -e

# Using CIFAR convolution instances from oval21/instances.csv
# Format: data/verification/oval21/{onnx_path},{vnnlib_path}

# ====== CONV Mode Tests - cifar_base_kw instances ======

# CONV with fixed scale (1.0, -1.0) - Default scales
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img8194-eps0.018300653594771243.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with fixed scale (0.5, -0.5)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img2578-eps0.021176470588235297.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 0.5 -0.5

# CONV with fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img4763-eps0.024705882352941175.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with fixed scale (10.0, -10.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img9410-eps0.043137254901960784.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 10.0 -10.0

# CONV with random scale (0.1, 100.0) - Default random range
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img8095-eps0.010457516339869282.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img8717-eps0.011633986928104577.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# CONV with random scale (0.01, 1000.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img6767-eps0.020000000000000004.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.01 1000.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img3161-eps0.018562091503267972.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with LeakyReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img3568-eps0.030457516339869282.vnnlib --mode conv --activation leakyrelu --scale_strat random --random_scale_range 0.1 100.0

# CONV with PReLU and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img550-eps0.01843137254901961.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img1781-eps0.04993464052287582.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with PReLU and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img8483-eps0.042091503267973854.vnnlib --mode conv --activation prelu --scale_strat random --random_scale_range 0.1 100.0

# ====== CONV Mode Tests - cifar_wide_kw instances ======

# CONV with fixed scale (1.0, -1.0) on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with random scale (0.1, 100.0) on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img7564-eps0.04183006535947713.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0) on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img8517-eps0.006928104575163399.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0) on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img2732-eps0.04967320261437909.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with random scale (0.5, 50.0) on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img1909-eps0.0033986928104575162.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV Mode Tests - cifar_deep_kw instances ======

# CONV with fixed scale (1.0, -1.0) on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img4405-eps0.036732026143790855.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with random scale (0.1, 100.0) on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img7878-eps0.009934640522875817.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU and fixed scale (1.0, -1.0) on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img1052-eps0.010718954248366015.vnnlib --mode conv --activation leakyrelu --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU and fixed scale (2.0, -2.0) on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img4325-eps0.01673202614379085.vnnlib --mode conv --activation prelu --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with random scale (0.5, 50.0) on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img4168-eps0.03150326797385621.vnnlib --mode conv --activation relu --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV with stable_tau_strat variations ======

# CONV with stable_tau_strat (big) on base network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with stable_tau_strat (small) on base network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with stable_tau_strat (big) and random scales on wide network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img5989-eps0.028235294117647063.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0

# CONV with stable_tau_strat (small) and random scales on deep network
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img8366-eps0.0550326797385621.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0

# ====== CONV with stable_tau_strat and -n parameter variations ======

# CONV with stable_tau_strat (big) on base network with -n 0
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# CONV with stable_tau_strat (big) on base network with -n 2
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 2

# CONV with stable_tau_strat (big) on base network with -n 5
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat fixed --fixed_scale 1.0 -1.0 -n 5

# CONV with stable_tau_strat (small) on base network with -n 0
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 0

# CONV with stable_tau_strat (small) on base network with -n 3
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img876-eps0.024836601307189544.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat fixed --fixed_scale 1.0 -1.0 -n 3

# CONV with stable_tau_strat (big) and random scales on wide network with -n 2
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img5989-eps0.028235294117647063.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 2

# CONV with stable_tau_strat (big) and random scales on wide network with -n 4
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img5989-eps0.028235294117647063.vnnlib --mode conv --activation relu --stable_tau_strat big --scale_strat random --random_scale_range 0.5 50.0 -n 4

# CONV with stable_tau_strat (small) and random scales on deep network with -n 0
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img8366-eps0.0550326797385621.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0 -n 0

# CONV with stable_tau_strat (small) and random scales on deep network with -n 4
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img8366-eps0.0550326797385621.vnnlib --mode conv --activation relu --stable_tau_strat small --scale_strat random --random_scale_range 0.1 100.0 -n 4

# ====== CONV with LeakyReLU alpha parameter tests ======

# CONV with LeakyReLU (alpha=0.01) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img8194-eps0.018300653594771243.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU (alpha=0.1) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img2578-eps0.021176470588235297.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with LeakyReLU (alpha=0.2) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.2 --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with LeakyReLU (alpha=0.01) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img4405-eps0.036732026143790855.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.01 --scale_strat random --random_scale_range 0.1 100.0

# CONV with LeakyReLU (alpha=0.1) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img4763-eps0.024705882352941175.vnnlib --mode conv --activation leakyrelu --leakyrelu_alpha 0.1 --scale_strat random --random_scale_range 0.5 50.0

# ====== CONV with PReLU slope_range parameter tests ======

# CONV with PReLU (slope=0.01-0.25) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img7564-eps0.04183006535947713.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU (slope=0.1-0.5) and fixed scale (1.0, -1.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img7878-eps0.009934640522875817.vnnlib --mode conv --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat fixed --fixed_scale 1.0 -1.0

# CONV with PReLU (slope=0.01-0.1) and fixed scale (2.0, -2.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_base_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_base_kw-img9410-eps0.043137254901960784.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.1 --scale_strat fixed --fixed_scale 2.0 -2.0

# CONV with PReLU (slope=0.01-0.25) and random scale (0.1, 100.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_wide_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_wide_kw-img8517-eps0.006928104575163399.vnnlib --mode conv --activation prelu --prelu_slope_range 0.01 0.25 --scale_strat random --random_scale_range 0.1 100.0

# CONV with PReLU (slope=0.1-0.5) and random scale (0.5, 50.0)
python anywhere_main.py --verbosity 10 split  --net data/verification/oval21/onnx/cifar_deep_kw.onnx  --spec data/verification/oval21/vnnlib/cifar_deep_kw-img1052-eps0.010718954248366015.vnnlib --mode conv --activation prelu --prelu_slope_range 0.1 0.5 --scale_strat random --random_scale_range 0.5 50.0
