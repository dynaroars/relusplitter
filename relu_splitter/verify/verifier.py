import subprocess
import sys
import logging
import os


RESMONITOR_PATH = "libs/resmonitor.py"


TimeoutError = "timeout"
OutOfMemoryError = "memout"
ModelNotFoundError = "instance_not_generated"
VnnlibNotFoundError = "vnnlib_not_generated"
UnknownError = "error"
VerifiedResult = "unsat"
FalsifiedResult = "sat"
UnknownResult = "unknown"



class Verifier:
    logger = logging.getLogger(__name__)

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    @classmethod
    def _gen_prog(cls, prog_conf):
        raise NotImplementedError("_gen_prog not implemented in sub class {cls.__name__}")

    @classmethod
    def gen_prog(cls, prog_conf):
        ram  = prog_conf.get('ram', None)
        time = prog_conf.get('timeout', None)
        verbosity = prog_conf.get('verbosity', 1)
        prog = cls._gen_prog(prog_conf)


        cmd = f"python3 {RESMONITOR_PATH}"

        if ram is not None:
            cmd += f" -M {ram}"
        if time is not None:
            cmd += f" -T {time}"
        if verbosity == 0:
            cmd += " -q"
        elif verbosity == 1:
            cmd += " -v"

        cmd += f" {prog}"

        return cmd

    @classmethod
    def execute(cls, prog_conf):
        cmd = cls.gen_prog(prog_conf)
        log_path = prog_conf.get('log_path')
        cls.logger.debug(f"config: {prog_conf}")
        cls.logger.info(f"Executing verification ... log path: {log_path}")
        cls.logger.info(cmd)

        with open(log_path, "w") as veri_log_fp:
            sp = subprocess.Popen(cmd, shell=True, stdout=veri_log_fp, stderr=veri_log_fp)

        cls.logger.debug(f"Verification process pid: {sp.pid}, waiting...")
        sp.wait()
        if sp.returncode != 0:
            cls.logger.error(f"Verification failed with return code {sp.returncode}")
            cls.logger.error(f"Verification log path: {log_path}")
            return None
        else:
            cls.logger.info("Verification finished successfully")
            return cls.analyze(prog_conf)
    @classmethod
    def _analyze(cls, lines):
        raise NotImplementedError("_analyze not implemented in sub class {cls.__name__}")
    
    @classmethod
    def pre_analyze(cls, prog_conf):
        veri_ans = None
        veri_time = None

        with open(prog_conf['log_path'], "r") as fp:
            lines = fp.readlines()

        # TODO: fix this with resmonitor -lli
        # for now just get the timeout value from the config
        for l in lines:
            if "Timeout (terminating process)" in l:
                veri_ans = "timeout"
                # veri_time = float(l.strip().split()[-1])
                veri_time = prog_conf['timeout']
            elif "Out of Memory" in l:
                veri_ans = "memout"
                # veri_time = float(l.strip().split()[-1])
                veri_time = prog_conf['timeout']

        return veri_ans, veri_time

    @classmethod
    def analyze(cls, prog_conf):
        log_path = prog_conf.get('log_path')

        with open(log_path, "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = cls.pre_analyze(prog_conf)
        if veri_ans is not None:
            return veri_ans, veri_time
        else:
            return cls._analyze(lines)

        if veri_ans in [FalsifiedResult, VerifiedResult, UnknownResult]:
            pass
        elif veri_ans == "timeout":
            veri_ans = TimeoutError
        elif veri_ans == "out_of_memory":
            veri_ans = OutOfMemoryError
        elif veri_ans == "model_not_exist":
            veri_ans = ModelNotFoundError
        elif veri_ans == "vnnlib_not_exist":
            veri_ans = VnnlibNotFoundError
        elif veri_ans == "error":
            veri_ans = UnknownError
        elif veri_ans is None:
            cls.logger.error(f"veri_ans is None in {log_path}")
            veri_ans = UnknownError
        else:
            cls.logger.error(f"unknown veri_ans: {veri_ans} in {log_path}")
            veri_ans = UnknownError

        
        if veri_time is None:
            cls.logger.error(f"veri_time is None in {log_path}")
            print(veri_ans)

        if veri_ans != 'timeout' and veri_time > prog_conf['timeout']:
            veri_ans = 'timeout'
            veri_time = prog_conf['timeout']

        return veri_ans, veri_time   

