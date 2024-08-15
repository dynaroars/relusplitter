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
benchmarks = {
    'acasxu': acasxu,
    'mnist_fc': mnist_fc,
}