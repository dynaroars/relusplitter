# cifar
python main.py split  --net /home/lli/storage/vnncomp2022_benchmarks/benchmarks/oval21/onnx/cifar_base_kw.onnx --spec /home/lli/storage/vnncomp2022_benchmarks/benchmarks/oval21/vnnlib/cifar_base_kw-img2578-eps0.021176470588235297.vnnlib --verify neuralsat  --input_shape 1 3 32 32

# vgg - doesnt work so well, the specs are stange. for most inputs lb=ub