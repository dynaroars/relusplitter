SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT=$SCRIPT_DIR/..

INPUT_DIR=$TOOL_ROOT/Seed_Inputs

echo $INPUT_DIR

cd $INPUT_DIR
git clone git@github.com:ChristopherBrix/vnncomp2022_benchmarks.git
cd vnncomp2022_benchmarks/benchmarks
mv sri_resnet_a ../..
mv sri_resnet_b ../..
mv oval21 ../..
mv rl_benchmarks ../..
mv cifar2020 ../..
mv mnist_fc ../..



cd $INPUT_DIR
git clone git@github.com:ChristopherBrix/vnncomp2023_benchmarks.git
cd vnncomp2023_benchmarks/benchmarks
mv collins_rul_cnn ../..
mv tllverifybench ../..
mv collins_yolo_robustness ../..

cd $INPUT_DIR
rm -rf vnncomp2022_benchmarks
rm -rf vnncomp2023_benchmarks
gunzip */onnx/* */vnnlib/*

