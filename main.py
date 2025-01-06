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

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the split command
    split_parser = subparsers.add_parser('split', help='split command help')

    # shared parameters
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--output', type=str, default='splitted.onnx', help='Output path for the new model')
    split_parser.add_argument('--input_shape', type=int, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')


    split_parser.add_argument('--split_idx', type=int, default=0, help='Index of the layer to split')
    
    split_parser.add_argument('--mask', type=str, default='stable+', help='Mask for splitting',
                              choices=['stable+', 'stable-', 'stable', 'unstable', 'all', 'unstable_n_stable+'])
    split_parser.add_argument('--n_splits', type=int, default=None, help='Number of splits (strict), this will override min_splits and max_splits')
    split_parser.add_argument('--min_splits', type=int, default=1, help='Minimum number of splits')
    split_parser.add_argument('--max_splits', type=int, default=sys.maxsize, help='Maximum number of splits')
    split_parser.add_argument('--scale_factor', type=float, nargs=2, default=[1.0,-1.0], help='Scale factor for the')
    # conv parameters
    split_parser.add_argument('--conv_strategy', type=str, default='random', help='Splitting strategy',
                              choices=['single', 'random', 'reluS+', 'reluS-', 'adaptive'])
    # fc parameters
    split_parser.add_argument('--fc_strategy', type=str, default='random', help='Splitting strategy',
                              choices=['single', 'random', 'reluS+', 'reluS-', 'adaptive'])

    



    
    

    
    
    split_parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')
    split_parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for closeness check')
    split_parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for closeness check')
    split_parser.add_argument('--device', type=str, default=default_device, help='Device for the model closeness check',)

    split_parser.add_argument('--verify', type=str, default=False, help='run verification with verifier',
                              choices=['neuralsat', 'abcrown', 'marabou', 'nnenum'])

    # Subparser for the info command
    info_parser = subparsers.add_parser('info', help='Info command help')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    info_parser.add_argument('--spec', type=str, required=False, help='Path to the VNNLIB file')
    info_parser.add_argument('--input_shape', type=int, required=False, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')


    # Subparser for the baseline command
    baseline_parser = subparsers.add_parser('baseline', help='baseline command help')
    baseline_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    baseline_parser.add_argument('--output', type=str, default='baseline.onnx', help='Output path for the new model')
    baseline_parser.add_argument('--n_splits', type=int, default=1, help='Number of splits')
    baseline_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')
    baseline_parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for closeness check')
    baseline_parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for closeness check')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if args.command == 'split':
        onnx_path = Path(args.net)
        spec_path = Path(args.spec)
        output_path = Path(args.output)
        input_shape = args.input_shape

        conf = {
            'split_mask': args.mask,
            'fc_strategy': args.fc_strategy,
            'conv_strategy': args.conv_strategy,
            'min_splits': args.min_splits,
            'max_splits': args.max_splits,
            'n_splits': args.n_splits,
            'split_idx': args.split_idx,
            'scale_factor': args.scale_factor,
            'random_seed': args.seed,
            'atol': args.atol,
            'rtol': args.rtol,
            'device': args.device
        }
        if args.n_splits is not None:
            conf['min_splits'] = args.n_splits
            conf['max_splits'] = args.n_splits
        relusplitter = ReluSplitter(onnx_path, 
                                    spec_path, 
                                    input_shape = input_shape,
                                    logger = logger, 
                                    conf = conf)
        logger.info(f'Start splitting...')
        logger.info(f'Conf: {conf}')
        new_model = relusplitter.split(args.split_idx, conf)
        new_model.save(output_path)
        logger.info(f'Model saved to {output_path}')

        if args.verify:
            verifier = init_verifier(args.verify)
            verifier.set_logger(logger)
            # abc_conf_path = "/home/lli/tools/relusplitter/experiment/config/mnistfc.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp23/tllVerifyBench.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/experiment/config/reach_probability.yaml"
            abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/oval22.yaml"
            conf1 = {
                'onnx_path': onnx_path,
                'vnnlib_path': spec_path,
                'log_path': Path('veri_1.log'),
                'verbosity': 1,
                'num_workers': 10,
                'config_path': abc_conf_path,
            }
            conf2 = {
                'onnx_path': output_path,
                'vnnlib_path': spec_path,
                'log_path': Path('veri_2.log'),
                'verbosity': 1,
                'num_workers': 10,
                'config_path': abc_conf_path,
            }
            logger.info(f'Start verification using {args.verify}')
            print("Original instance:")
            print(colored(verifier.execute(conf1), 'green'))
            print("Splitted instance:")
            print(colored(verifier.execute(conf2), 'yellow'))
        logger.info(f'=== Done ===')

    elif args.command == 'info':
        if args.spec is not None:
            ReluSplitter.info(Path(args.net), Path(args.spec), args.input_shape)
        else:
            ReluSplitter.info_net_only(Path(args.net))
    elif args.command == 'baseline':
        onnx_path = Path(args.net)
        output_path = Path(args.output)
        new_model = ReluSplitter.get_fc_baseline_split(onnx_path, args.n_splits, args.split_idx, args.atol, args.rtol)
        new_model.save(output_path)
    else:
        parser.print_help()
        sys.exit(1)



