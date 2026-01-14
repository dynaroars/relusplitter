#!/bin/bash
# set -e

python anywhere_main.py --verbosity 10 info  --net stuff/cifar_base_kw.onnx  


passed=0
failed=0

python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 0 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Test 1 passed."
    passed=$((passed + 1))
else
    echo "Test 1 failed."
    failed=$((failed + 1))
fi
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode gemm --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Test 2 passed."
    passed=$((passed + 1))
else
    echo "Test 2 failed."
    failed=$((failed + 1))
fi
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 2 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Test 3 passed."
    passed=$((passed + 1))
else
    echo "Test 3 failed."
    failed=$((failed + 1))
fi
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode conv --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 3 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Test 4 passed."
    passed=$((passed + 1))
else
    echo "Test 4 failed."
    failed=$((failed + 1))
fi

python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode all --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 0 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Test 5 passed."
    passed=$((passed + 1))
else
    echo "Test 5 failed."
    failed=$((failed + 1))
fi
python anywhere_main.py --verbosity 10 split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib --mode all --activation relu --scale_strat fixed --fixed_scale 1.0 -1.0 --split_idx 3 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Test 6 passed."
    passed=$((passed + 1))
else
    echo "Test 6 failed."
    failed=$((failed + 1))
fi

echo "Total tests passed: $passed"
echo "Total tests failed: $failed"

rm splitted.onnx
