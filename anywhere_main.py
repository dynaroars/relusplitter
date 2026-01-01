from pathlib import Path
from relu_splitter.anywhere import ReluSplitter_Anywhere
import sys

onnx_path = "/home/lli/tools/relusplitter/data/mnist_fc/onnx/mnist-net_256x4.onnx"
vnnlib_path = "/home/lli/tools/relusplitter/data/mnist_fc/vnnlib/prop_1_0.03.vnnlib"

if len(sys.argv) > 1:
    # override using provided onnx model
    onnx_path = sys.argv[1]



bruh = ReluSplitter_Anywhere(onnx_path, vnnlib_path)
# print([n.name for n in bruh.get_splittable_nodes()])

conf = {
    "split_activation": "prelu",
    "equiv_chk_conf": {
        # "n": 100,
        # "atol": 1e-5,
        # "rtol": 1e-5
    },
    "candidate_selection_conf": {
        "split_mask": "stable+"
    },
    "param_selection_conf":{
    }
    

}

new_model, baseline = bruh.split("Gemm", 1, conf)
new_model.save(Path("anywhere_test.onnx"))