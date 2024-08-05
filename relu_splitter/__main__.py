import torch
import onnx
import logging
import argparse
import sys

import torch.nn as nn
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .core import ReluSplitter

from .utils.logger import get_logger



def parse_args():
    parser = argparse.ArgumentParser(description="Parser for network verification arguments")
    parser.add_argument('--verbosity', type=int, default=10, help='Verbosity level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)')

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the main command
    split_parser = subparsers.add_parser('split', help='Main command help')
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    split_parser.add_argument('--split_strategy', type=str, default='single', help='Splitting strategy')
    split_parser.add_argument('--max_splits', type=str, default=None, help='Maximum number of splits')
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')

    # Subparser for the info command
    info_parser = subparsers.add_parser('info', help='Info command help')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')

    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()

    logging.basicConfig(level=args.verbosity, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    if args.command == 'split':
        onnx_path = args.net
        spec_path = args.spec
        random_seed = 0
        max_splits = None
        split_idx = 0
        conf = {
            # "logger": logger,
            "random_seed": 0,
            "split_strategy": "single",
            "max_splits": None,
            "split_idx": 0
        }
        relusplitter = ReluSplitter(onnx_path, spec_path, conf=conf)
        # relusplitter.analyze_model()
        relusplitter.split()
    elif args.command == 'info':
        onnx_path = args.net




if __name__ == '__main__':
    main()