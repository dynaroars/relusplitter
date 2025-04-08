import os
import subprocess
from pathlib import Path


TOOL_NAME = "ReluSplitter"
TOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(TOOL_ROOT, "lib")
ENVS_PATH = os.path.join(os.environ.get("TOOL_ROOT"), ".envs")
PYTHON_EXE = os.path.join(ENVS_PATH, TOOL_NAME, "bin", "python")

INPUT_DIR = os.path.join(TOOL_ROOT, "Seed_Inputs")
SELECTED_INSTANCES = os.path.join(INPUT_DIR, "selected_instances.csv")

OUTPUT_DIR = os.path.join(TOOL_ROOT, "Generated_Instances")
ONNX_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "onnx")
VNNLIB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "vnnlib")
GENERATED_INSTANCES = os.path.join(OUTPUT_DIR, "generated_instances.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
os.makedirs(VNNLIB_OUTPUT_DIR, exist_ok=True)

os.environ["TOOL_NAME"] = TOOL_NAME
os.environ["TOOL_ROOT"] = TOOL_ROOT
os.environ["LIB_PATH"] = os.path.join(TOOL_ROOT, "lib")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        os.environ.get("PYTHONPATH", ""),
        TOOL_ROOT,
        LIB_PATH,
    ]
)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


with open(SELECTED_INSTANCES, "r") as f, open(GENERATED_INSTANCES, "w") as g:
    for line in f:
        onnx, vnnlib, timeout, mode, split_idx, n_splits = [col.strip() for col in line.split(",")]

        onnx_stem, vnnlib_stem = Path(onnx).stem, Path(vnnlib).stem
        output_onnx_path = f"{ONNX_OUTPUT_DIR}/{onnx_stem}-{vnnlib_stem}-M{mode}-S{split_idx}-N{n_splits}.onnx"
        output_vnnlib_path = f"{VNNLIB_OUTPUT_DIR}/{onnx_stem}-{vnnlib_stem}-M{mode}-S{split_idx}-N{n_splits}.vnnlib"
        output_timeout = timeout

        GENERATED_INSTANCES.write(f"{output_onnx_path},{output_vnnlib_path},{output_timeout}\n")




# run 
# .envs/ReluSplitter/bin/python main.py  split --net ../relusplitter/data/oval21/onnx/cifar_wide_kw.onnx  --spec ../relusplitter/data/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib --closeness_check  --mode conv

# subprocess.run(
#     [
#         PYTHON_EXE,
#         os.path.join(os.environ["TOOL_ROOT"], "main.py"),
#         "split",
#         "--net",
#         "../relusplitter/data/oval21/onnx/cifar_wide_kw.onnx",
#         "--spec",
#         "../relusplitter/data/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib",
#         "--closeness_check",
#         "--mode",
#         "conv",
#     ],
# )

