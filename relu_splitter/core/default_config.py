import sys


default_config = {
    'seed': 0,
    'mask': 'stable',
    'split_idx': 0,
    'split_strategy': 'single',
    'max_splits': sys.maxsize,
    'atol': 1e-5,
    'rtol': 1e-5,
    'output': "splitted.onnx"
}
