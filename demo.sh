

# info
python main.py info --net data/acasxu_converted/onnx/ACASXU_run2a_1_1_batch_2000_converted.onnx
nsat --net data/acasxu_converted/onnx/ACASXU_run2a_1_1_batch_2000_converted.onnx --spec data/acasxu_converted/vnnlib/prop_3.vnnlib

# split
python main.py split --net data/acasxu_converted/onnx/ACASXU_run2a_1_1_batch_2000_converted.onnx --spec data/acasxu_converted/vnnlib/prop_3.vnnlib --seed 0 --output acasxu_test_iter1.onnx

nsat --net acasxu_test_iter1.onnx --spec data/acasxu_converted/vnnlib/prop_3.vnnlib
