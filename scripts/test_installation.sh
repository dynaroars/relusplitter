python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy reluS-  --mask unstable --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy reluS-  --mask stable-  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy reluS+  --mask unstable --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy reluS+  --mask stable+  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0



python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy single  --mask stable+  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy single  --mask stable-  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy single  --mask unstable --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy random  --mask stable+  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy random  --mask stable-  --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0
python main.py --verbosity 20  split  --net data/mnist_fc/onnx/mnist-net_256x4.onnx --spec data/mnist_fc/vnnlib/prop_4_0.03.vnnlib   --split_strategy random  --mask unstable --n_splits 5  --seed 4 --verify neuralsat  --split_idx 0