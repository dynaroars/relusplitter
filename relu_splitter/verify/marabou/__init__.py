import pathlib
from ..verifier import Verifier


class Marabou(Verifier):
    name = "marabou"
    @classmethod
    def _gen_prog(cls, prog_conf):
        model_path = prog_conf.get('onnx_path')
        property_path = prog_conf.get('vnnlib_path')

        cmd = f"{pathlib.Path(__file__).parent}/run_marabou.sh {model_path} {property_path} "

        num_workers = prog_conf.get('num_workers', None)
        timeout = prog_conf.get('timeout', None)
        milp = prog_conf.get('milp', False)
        snc = prog_conf.get('snc', False)

        if num_workers is not None:
            cmd += f" --num-workers {num_workers} "
        if timeout is not None:
            cmd += f" --timeout {timeout} "
        if milp == True:
            cmd += " --milp "
        if snc == True:
            cmd += " --snc "
        
        return cmd

    @classmethod
    @property
    def relavent_configs(cls):
        return ['num_workers', 'timeout', 'milp', 'snc']

    @classmethod
    def _analyze(cls, lines):
        veri_ans = None
        veri_time = None

        for l in lines:
            if l == "unsat\n":
                veri_ans = "unsat"
            elif l == "sat\n":
                veri_ans = "sat"
            elif "(resmonitor) Process finished successfully" in l:
                veri_time = float(l.strip().split()[-3][:-2])

        return veri_ans, veri_time