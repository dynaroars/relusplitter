from pathlib import Path
from relu_splitter.anywhere import ReluSplitter_Anywhere
import sys
from pathlib import Path
import torch
import argparse

from relu_splitter.utils.logger import default_logger as logger

# # onnx_path = "/home/lli/tools/relusplitter/data/verification/sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx"
# # vnnlib_path = "/home/lli/tools/relusplitter/data/verification/sri_resnet_a/vnnlib/cifar10_spec_idx_232_eps_0.00350.vnnlib"

# onnx_path = "/home/lli/tools/relusplitter/data/verification/oval21/onnx/cifar_base_kw.onnx"
# vnnlib_path = "/home/lli/tools/relusplitter/data/verification/oval21/vnnlib/cifar_base_kw-img423-eps0.021960784313725494.vnnlib"

# # onnx_path = "/home/lli/tools/relusplitter/data/verification/oval21/onnx/cifar_base_kw.onnx"
# # vnnlib_path = "/home/lli/tools/relusplitter/data/verification/oval21/vnnlib/cifar_base_kw-img423-eps0.021960784313725494.vnnlib"


# # onnx_path = "/home/lli/tools/relusplitter/data/mnist_fc/onnx/mnist-net_256x6.onnx"
# # vnnlib_path = "/home/lli/tools/relusplitter/data/mnist_fc/vnnlib/prop_11_0.03.vnnlib"

# if len(sys.argv) > 1:
#     # override using provided onnx model
#     # onnx_path = sys.argv[1]
#     n_split = int(sys.argv[1])



# bruh = ReluSplitter_Anywhere(onnx_path, vnnlib_path)
# # print([n.name for n in bruh.get_splittable_nodes()])

# conf = {
#     "seed": 42,
#     "split_activation": "relu",
#     "equiv_chk_conf": {
#         # "n": 100,
#         # "atol": 1e-5,
#         # "rtol": 1e-5
#     },
#     "n_splits": n_split,
#     "param_selection_conf":{
#     }
    

# }

# new_model, baseline = bruh.split("Conv", 0, conf)
# # new_model, baseline = bruh.split("Gemm", 0, conf)
# new_model.save(Path("anywhere_test.onnx"))

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_parser():
    parser = argparse.ArgumentParser(description='ReLUSplitter (Anywhere)')
    parser.add_argument("--verbosity", type=int, default=20, help="Logging verbosity level (10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR)")

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    info_parser = subparsers.add_parser('info', help='Display model information')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')

    split_parser = subparsers.add_parser('split', help='Split ReLU activations in the model')
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--output', type=str, default='splitted.onnx', help='Output path for the new model')
    split_parser.add_argument('--baseline', type=str, default=None, help='Output path for the baseline model')
    split_parser.add_argument('--input_shape', type=int, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')
    split_parser.add_argument('--mode', type=str, default='all', help='Mode for splitting',
                              choices=['gemm', 'conv', 'all'])
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index of the layer to split')
    split_parser.add_argument('-n', type=int, default=None, help='Number of destablizing splits')
    split_parser.add_argument('--seed', type=int, default=35, help='Seed for random number generation')
    split_parser.add_argument('--activation', type=str, default='relu', help='Activation function to split',
                                choices=['relu', 'leaky_relu', 'prelu'])
    
    split_parser.add_argument('--closeness_check', action='store_true', default=True, help='Enable closeness check between original and splitted models')

    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if args.command == 'info':
        rsa = ReluSplitter_Anywhere(args.net, None)
        rsa.display_model_info()

    elif args.command == 'split':
        rsa = ReluSplitter_Anywhere(args.net, args.spec, input_shape=args.input_shape)
        conf = {
            "seed": args.seed,
            "split_activation": args.activation,
            "n_splits": args.n,
            "create_baseline": args.baseline is not None,
            "sorting_strat": "random",
        }
        new_model, baseline = rsa.split(args.mode, args.split_idx, conf)
        new_model.save(Path(args.output))
        print(f"Saved splitted model to {args.output}")
        if baseline is not None and args.baseline is not None:
            baseline.save(Path(args.baseline))
            print(f"Saved baseline model to {args.baseline}")
        
        if args.closeness_check:
            from relu_splitter.utils.onnx_utils import check_models_closeness
            closeness_results = check_models_closeness(
                rsa.model,
                [new_model, baseline],
                rsa.input_shape,
                device=default_device,
                n=10,
                atol=1e-6,
                rtol=1e-6
            )
            logger.info(f"Performed closeness check with 10 random samples on device {default_device}.")
            logger.info(f"Closeness results [Split]: {closeness_results[0]}")
            logger.info(f"Closeness results [Baseline]: {closeness_results[1]}")
            assert closeness_results[0][0] and closeness_results[1][0], "Closeness check failed!"

    else:
        parser.print_help()



