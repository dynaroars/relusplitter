import torch
import onnx
import logging
import argparse
import sys

import torch.nn as nn
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from relu_splitter.core import ReluSplitter


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
    split_parser.add_argument('--split_strategy', type=str, default='single', help='Splitting strategy')
    split_parser.add_argument('--max_splits', type=str, default=None, help='Maximum number of splits')
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')

    # Subparser for the info command
    info_parser = subparsers.add_parser('info', help='Info command help')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')

    return parser

def main():
    
    parser = get_parser()
    args = parser.parse_args()
    logger = setup_logging(args.verbosity)

    if args.command == 'split':
        onnx_path = args.net
        spec_path = args.spec
        random_seed = args.seed
        max_splits = args.max_splits
        split_idx = args.split_idx
        
        relusplitter = ReluSplitter(onnx_path, spec_path, random_seed, logger)
        new_model = relusplitter.split(args.split_idx, args.max_splits, args.split_strategy)

    elif args.command == 'info':
        onnx_path = args.net
        ReluSplitter.info(onnx_path)
    else:
        parser.print_help()
        sys.exit(1)



if __name__ == '__main__':
    main()
    print("BRUH")
