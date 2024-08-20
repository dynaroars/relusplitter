import os
import sys
import subprocess
from pathlib import Path



if __name__=="__main__":
    path = sys.argv[1]

    path = Path(path)
    for f in path.iterdir():
        if f.suffix == ".onnx":
            subprocess.run(["python", "experiment/scripts/matmul_add_2_gemm.py", str(f), f"data/tllverifybench_converted/onnx/{f.stem}_converted.onnx"])
    
    print("Done")