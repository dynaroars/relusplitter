from pathlib import Path


from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier

veri_base_conf = {
    "timeout": 600,
    "onnx_path": "data/mnist_fc/onnx/mnist-net_256x2.onnx",
    "vnnlib_path": "/home/lli/tools/relusplitter/data/mnist_fc/vnnlib/prop_7_0.03.vnnlib",
    "log_path": "experiment/logs/verify.log",
}

def main():
    neuralsat = init_verifier("neuralsat")
    abcrown = init_verifier("abcrown")

    print(neuralsat.execute(veri_base_conf))
    print(abcrown.execute(veri_base_conf))
    





if __name__ == "__main__":
    main()