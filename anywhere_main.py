from pathlib import Path
from relu_splitter.anywhere import ReluSplitter_Anywhere



bruh = ReluSplitter_Anywhere("/home/lli/tools/relusplitter/data/mnist_fc/onnx/mnist-net_256x4.onnx", "/home/lli/tools/relusplitter/data/mnist_fc/vnnlib/prop_1_0.03.vnnlib")
print([n.name for n in bruh.get_splittable_nodes()])

conf = {
    "split_activation": "leakyrelu",
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

new_model, baseline = bruh.split("Gemm", 0, conf)
new_model.save(Path("anywhere_test.onnx"))