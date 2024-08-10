import torch
import onnx
import logging
import argparse
import sys
from pathlib import Path

import torch.nn as nn
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from relu_splitter.core import ReluSplitter
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.verify import init_verifier


def setup_logging(verbosity):
    logging.basicConfig(level=verbosity, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(verbosity)
    return logger

def get_parser():
    parser = argparse.ArgumentParser(description="Parser for network verification arguments")
    parser.add_argument('--verbosity', type=int, default=10, help='Verbosity level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)')

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the main command
    split_parser = subparsers.add_parser('split', help='Main command help')
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    split_parser.add_argument('--mask', type=str, default='stable', help='Mask for splitting')
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')
    split_parser.add_argument('--split_strategy', type=str, default='single', help='Splitting strategy')
    split_parser.add_argument('--max_splits', type=int, default=sys.maxsize, help='Maximum number of splits')
    
    split_parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for closeness check')
    split_parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for closeness check')

    split_parser.add_argument('--output', type=str, default="splitted.onnx", help='Output path for the new model')


    # Subparser for the info command
    info_parser = subparsers.add_parser('info', help='Info command help')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    info_parser.add_argument('--spec', type=str, required=False, help='Path to the VNNLIB file')

    # Subparser for the exp command
    exp_parser = subparsers.add_parser('exp', help='Exp command help')
    exp_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    exp_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    exp_parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    exp_parser.add_argument('--mask', type=str, default='stable', help='Mask for splitting')
    exp_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')
    exp_parser.add_argument('--split_strategy', type=str, default='single', help='Splitting strategy')
    exp_parser.add_argument('--max_splits', type=int, default=sys.maxsize, help='Maximum number of splits')
    
    exp_parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for closeness check')
    exp_parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for closeness check')


    return parser

def main():
    
    parser = get_parser()
    args = parser.parse_args()
    logger = setup_logging(args.verbosity)

    if args.command == 'split':
        onnx_path = Path(args.net)
        spec_path = Path(args.spec)
        output_path = Path(args.output)

        conf = {
            "split_mask": args.mask,
            "split_strategy": args.split_strategy,
            "max_splits": args.max_splits,
            "split_idx": args.split_idx,
            "random_seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol
        }
        
        relusplitter = ReluSplitter(onnx_path, 
                                    spec_path, 
                                    logger = logger, 
                                    conf = conf)
        new_model = relusplitter.split(args.split_idx)
        new_model.save(output_path)
        logger.info(f"Model saved to {output_path}")

    elif args.command == 'info':
        onnx_path = args.net
        spec_path = args.spec
        relusplitter = ReluSplitter(onnx_path, spec_path, logger=logger)
        if spec_path is not None:
            relusplitter.info()
        else:
            relusplitter.info_net_only()
    elif args.command == 'exp':
        verifier = init_verifier("neuralsat")
        onnx_path = Path(args.net)
        spec_path = Path(args.spec)

        conf_1 = {
            "split_mask": args.mask,
            "split_strategy": args.split_strategy,
            "max_splits": args.max_splits,
            "split_idx": args.split_idx,
            "random_seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol
        }
        conf_2 = conf_1.copy()
        conf_2["split_mask"] = "unstable"
        
        relusplitter_1 = ReluSplitter(onnx_path, 
                                    spec_path, 
                                    logger = logger, 
                                    conf = conf_1)
        relusplitter_2 = ReluSplitter(onnx_path, 
                                    spec_path, 
                                    logger = logger, 
                                    conf = conf_2)
        m1 = relusplitter_1.split(args.split_idx)
        m2 = relusplitter_2.split(args.split_idx)
        m1.save(Path("m1.onnx"))
        m2.save(Path("m2.onnx"))
        co = { "onnx_path": onnx_path, "vnnlib_path": spec_path, "log_path": "log.txt" }
        c1 = { "onnx_path": "m1.onnx", "vnnlib_path": spec_path, "log_path": "log.txt" }
        c2 = { "onnx_path": "m2.onnx", "vnnlib_path": spec_path, "log_path": "log.txt" }
        print(verifier.execute(co))
        print(verifier.execute(c1))
        print(verifier.execute(c2))
        
    else:
        parser.print_help()
        sys.exit(1)



if __name__ == '__main__':
    main()
    print("BRUH")
