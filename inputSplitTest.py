from relu_splitter.utils.logger import logger

import torch
import onnx
import argparse
import sys
from pathlib import Path
from termcolor import colored


from relu_splitter.core import ReluSplitter
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.verify import init_verifier


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_parser():
    parser = argparse.ArgumentParser(description='Parser for network verification arguments')
    parser.add_argument('--verbosity', type=int, default=20, help='Verbosity level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)')


    # shared parameters
    parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    print(args.net)
    print(args.spec)
    print(args.verbosity)