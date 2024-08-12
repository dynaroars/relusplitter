from relu_splitter.utils import logger

import torch
import onnx
import sys
from pathlib import Path

from relu_splitter.core import ReluSplitter
from relu_splitter.verify import init_verifier


acasxu = {
    "path"      : Path('data/acasxu'),
    "timeout"   : 300,
}

mnist_fc = {
    "path"      : Path('data/mnist_fc'),
    "timeout"   : 600,
}


def get_instances(benchmark):
    for 