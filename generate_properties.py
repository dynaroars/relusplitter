import os
import subprocess

TOOL_NAME = "ReluSplitter"
TOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(TOOL_ROOT, "lib")
ENVS_PATH = os.path.join(os.environ.get("TOOL_ROOT"), ".envs")
PYTHON_EXE = os.path.join(ENVS_PATH, TOOL_NAME, "bin", "python")


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



# run 
# .envs/ReluSplitter/bin/python main.py  split --net ../relusplitter/data/oval21/onnx/cifar_wide_kw.onnx  --spec ../relusplitter/data/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib --closeness_check  --mode conv

subprocess.run(
    [
        PYTHON_EXE,
        os.path.join(os.environ["TOOL_ROOT"], "main.py"),
        "split",
        "--net",
        "../relusplitter/data/oval21/onnx/cifar_wide_kw.onnx",
        "--spec",
        "../relusplitter/data/oval21/vnnlib/cifar_wide_kw-img6432-eps0.034771241830065365.vnnlib",
        "--closeness_check",
        "--mode",
        "conv",
    ],
)

