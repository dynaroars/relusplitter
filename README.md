

# ReluSplitter

ReluSplitter is a DNN verification benchmark generation tool for fully-connected and convolutional networks. It takes input as a DNN verification problem instance (network + property) and generates a modified network s.t. the modified network carries the same semantic of the orginal network, but it is harder to verify the property on the modified network. ReluSplitter achieves this by systematically de-stablizing neurons in the network. 

![Overview](stuff/figs/tool_overview.PNG)


## Features
Currently, our tool accepts input in the standard `onnx` and `vnnlib` format, and
supports two common layer types.
- Fully-connected layer (onnx.GEMM)
- Convolutional layer (onnx.CONV)

Technically, any network with above layers can be splitted. However, you might encounter compatibility issues due to factors like ONNX version differences or complex network structures. If you believe your input should work but doesn't, feel free to [open an issue](#) or contact us.


### Why ReluSplitter?

- ✅ **Semantic-preserving**: The generated network always retains the same semantics as the original. For example, if the original instance is `UNSAT`, the modified instance will also be `UNSAT`.  
- ⚡ **Fast generation**: A new DNNV instance can be created within seconds—significantly faster than alternative benchmark generation approaches that require training or distillation.


## Installation
- install `ReluSplitter`
```bash
./scripts/install.sh
```

## Usage
**Important:** Before running any commend below, first run `source rdy.sh` to activate the enviroment.


basic usage
- info
    ```bash
    python anywhere_main.py info  --net stuff/cifar_base_kw.onnx  
    ==================== Model Info ====================
    Model: stuff/cifar_base_kw.onnx
    Model input shapes: {'input.1': [1, 3, 32, 32]}
    --------------------------------------------------------------------------------
    idx        | name                           | op_type    | layer_size          
    --------------------------------------------------------------------------------
            0 | Conv_0                         | Conv       | torch.Size([8])     
            1 | Conv_2                         | Conv       | torch.Size([16])    
            2 | Gemm_5                         | Gemm       | torch.Size([100])   
            3 | Gemm_7                         | Gemm       | torch.Size([10])    
    ```
- split (FC)
    ```bash
    python anywhere_main.py split  --net stuff/ACASXU_run2a_2_2_batch_2000_converted.onnx  --spec stuff/prop_4.vnnlib
    ```
    Output:
    ```bash
    INFO     15:50:05     Splitting node Gemm_2 of type Gemm at index 0 with config: {'seed': 35, 'split_activation': 'relu', 'n_splits': None, 'create_baseline': False, 'candidiate_strat': 'random', 'param_conf': {'gemm_tau_strat': 'random', 'stable_tau_strat': 'random', 'stable_tau_margin': (10.0, 50.0), 'split_scale_strat': 'fixed', 'fixed_scales': (1.0, -1.0), 'random_scale_range': (0.1, 100.0)}, 'additional_activation_conf': {'leakyrelu_alpha': 0.01, 'prelu_slope_range': (0.01, 0.25)}}
    /home/lli/tools/relusplitter/libs/onnx2pytorch/convert/model.py:180: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    if not self.experimental and inputs[0].shape[self.batch_dim] > 1:
    INFO     15:50:05     Splitting gemm neurons: [24, 47, 31, 40, 14, 12, 28, 42, 44, 4, 13, 43, 20, 22, 45, 38, 19, 5, 2, 25, 7, 26, 1, 29, 46, 10, 39, 30, 11, 15, 49, 41, 34, 32, 0, 37, 6, 23, 17, 33, 3, 36, 16, 27, 18, 9, 48, 8, 21, 35]...
    Constructing new ReLU layers: 100%|████████████████████████████████████████████| 50/50 [00:00<00:00, 33989.50it/s]
    Saved splitted model to splitted.onnx
    INFO     15:50:05     Performed closeness check with 10 random samples on device cuda.
    INFO     15:50:05     Closeness results [Split]: (True, tensor(9.5367e-06, device='cuda:0', grad_fn=<MaxBackward1>))
    INFO     15:50:05     Closeness results [Baseline]: (True, None)
    ```
- split (Conv)
    ```bash
    python anywhere_main.py split  --net stuff/cifar_base_kw.onnx  --spec stuff/cifar_base_kw-img1340-eps0.020130718954248367.vnnlib
    ```
    Output:
    ```bash
    INFO     15:52:26     Splitting node Conv_0 of type Conv at index 0 with config: {'seed': 35, 'split_activation': 'relu', 'n_splits': None, 'create_baseline': False, 'candidiate_strat': 'random', 'param_conf': {'gemm_tau_strat': 'random', 'stable_tau_strat': 'random', 'stable_tau_margin': (10.0, 50.0), 'split_scale_strat': 'fixed', 'fixed_scales': (1.0, -1.0), 'random_scale_range': (0.1, 100.0)}, 'additional_activation_conf': {'leakyrelu_alpha': 0.01, 'prelu_slope_range': (0.01, 0.25)}}
    /home/lli/tools/relusplitter/libs/onnx2pytorch/convert/model.py:180: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    if not self.experimental and inputs[0].shape[self.batch_dim] > 1:
    INFO     15:52:26     Splitting conv filters: [0, 4, 3, 7, 2, 1, 6, 5]...
    Creating split Conv with ReLU: 100%|█████████████████████████████████████████████| 8/8 [00:00<00:00, 18265.89it/s]
    Saved splitted model to splitted.onnx
    INFO     15:52:26     Performed closeness check with 10 random samples on device cuda.
    INFO     15:52:26     Closeness results [Split]: (True, tensor(4.7684e-07, device='cuda:0', grad_fn=<MaxBackward1>))
    INFO     15:52:26     Closeness results [Baseline]: (True, None)
    ```

ReluSplitter also provide a number of parameters for users to explore. For example: the number of split, the type of layer, or the type of neuron to split (stably active, stably inactive, unstable). Run with `-h` flag to display avaliable options.



