import pathlib
from ..verifier import Verifier


class Marabou(Verifier):
    @classmethod
    def _gen_prog(cls, prog_conf):
        model_path = prog_conf.get('onnx_path')
        property_path = prog_conf.get('vnnlib_path')

        cmd = f"{pathlib.Path(__file__).parent}/run_marabou.sh {model_path} {property_path} "

        num_workers = prog_conf.get('num_workers', None)
        timeout = prog_conf.get('timeout', None)

        if num_workers is not None:
            cmd += f" --num-workers {num_workers} "
        if timeout is not None:
            cmd += f" --timeout {timeout} "
        
        return cmd

    @classmethod
    def _analyze(cls, lines):
        veri_ans = None
        veri_time = None

        for l in lines:
            if "AssertionError" in l:
                veri_ans = 'error'
                veri_time = -1
            elif "CUDA error: out of memory" in l:
                veri_ans = 'out_of_memory'
                veri_time = -1
            elif "CUDA out of memory" in l:
                veri_ans = 'out_of_memory'
                veri_time = -1
            elif "RuntimeError" in l:
                veri_ans = 'error'
                veri_time = -1

            elif "[!] Result:" in l:
                veri_ans = l.strip().split()[-1]
            elif "[!] Runtime:" in l:
                veri_time = float(l.strip().split()[-1])

        return veri_ans, veri_time