from pathlib import Path
import os

tool_root   = Path(os.environ["TOOL_ROOT"])
acasxu = {
    "name"      : "acasxu",   
    "path"      : Path(tool_root/'data'/'acasxu_converted'),
    "timeout"   : 120,
}
mnist_fc = {
    "name"      : "mnist_fc",
    "path"      : Path(tool_root/'data'/'mnist_fc'),
    "timeout"   : 300,
}
nn4sys = {
    "name"      : "nn4sys",
    "path"      : Path(tool_root/'data'/'nn4sys'),
    # "timeout"   : 300,
}
tll = {
    "name"      : "tllverifybench",
    "path"      : Path(tool_root/'data'/'tllverifybench_converted'),
    "timeout"   : 600,
}
collins_cnn = {
    "name"      : "collins_rul_cnn",
    "path"      : Path(tool_root/'data'/'collins_rul_cnn'),
    # "timeout"   : 300,
}

benchmarks = {
    'acasxu': acasxu,
    'mnist_fc': mnist_fc,
    'nn4sys': nn4sys,
    'collins_cnn': collins_cnn,
    'tll': tll,
}
