SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT=$SCRIPT_DIR/..

INPUT_DIR=$TOOL_ROOT/Seed_Inputs

echo $INPUT_DIR

# MNIST FC
# compatible with Relusplitter env
cd $INPUT_DIR
git clone https://github.com/pat676/mnist_fc_vnncomp2022.git

# OVAL21
# compatible with Relusplitter env
cd $INPUT_DIR
git clone https://github.com/alessandrodepalma/oval21-benchmark.git

# SRI ResNet A
# also compatible with Relusplitter env
cd $INPUT_DIR
git clone https://github.com/mnmueller/vnn_comp_22_bench_sri.git sri_resnet_a
cd sri_resnet_a
git checkout ResNetA

cd $INPUT_DIR
git clone https://github.com/mnmueller/vnn_comp_22_bench_sri.git sri_resnet_b
cd sri_resnet_b
git checkout ResNetB



# cd $INPUT_DIR
# git clone git@github.com:ChristopherBrix/vnncomp2022_benchmarks.git
# cd vnncomp2022_benchmarks/benchmarks
# mv sri_resnet_a ../..
# mv sri_resnet_b ../..
# mv oval21 ../..
# mv rl_benchmarks ../..
# mv cifar2020 ../..
# mv mnist_fc ../..



# cd $INPUT_DIR
# git clone git@github.com:ChristopherBrix/vnncomp2023_benchmarks.git
# cd vnncomp2023_benchmarks/benchmarks
# mv collins_rul_cnn ../..
# mv tllverifybench ../..
# mv collins_yolo_robustness ../..

# cd $INPUT_DIR
# rm -rf vnncomp2022_benchmarks
# rm -rf vnncomp2023_benchmarks
# gunzip */onnx/* */vnnlib/*

