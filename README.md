# NeurIPS25 Instruction


## installation
```bash
././scripts/install.sh
```

## usage
After installation, first activate the conda enviroment and export env variables via
```bash
source rdy.sh
```

Then run the tool via
```
python main.py
```


- Basic usage
    ```
    python main.py split --net <onnx_input.onnx>  --spec <vnnlib_input.vnnlib> --mode fc  --out <output_onnx.onnx>
    ```

- example
    ```
    python main.py split --net data/acasxu_converted/onnx/ACASXU_run2a_3_3_batch_2000_converted.onnx  --spec data/acasxu_converted/vnnlib/prop_1.vnnlib --mode fc --out test.onnx

    TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    if not self.experimental and inputs[0].shape[self.batch_dim] > 1:
    2025-05-14 07:44:04,987 - relu_splitter.utils.logger - INFO - fc_split.py:64 - ============= Split Mask Sumamry =============
    2025-05-14 07:44:04,987 - relu_splitter.utils.logger - INFO - fc_split.py:65 - stable+: 8       stable-: 20
    2025-05-14 07:44:04,987 - relu_splitter.utils.logger - INFO - fc_split.py:67 - unstable: 22     all: 50
    2025-05-14 07:44:04,988 - relu_splitter.utils.logger - INFO - fc_split.py:85 - Selecting 8/8 stable+ ReLUs to split
    UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
    w = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[w]) )
    Constructing new layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 50864.71it/s]
    2025-05-14 07:44:04,992 - relu_splitter.utils.logger - INFO - main.py:150 - Split model saved to test.onnx
    2025-05-14 07:44:04,993 - relu_splitter.utils.logger - INFO - main.py:173 - === Done ===
    ```


## Reproducing experiment

### download benchmarks
TODO

### install gurobi
```bash
./scripts/libs/install_gurobi.sh
```
After installation, place the gurobi license to `libs/gurobi.lic` 


### Install Verifiers
We provided script for installing the verifiers.

- alpha-beta-crown
    ````bash
    ./scripts/libs/install_abcrown.sh
- marabou
    ````bash
    ./scripts/libs/install_marabou.sh
- neuralsat
    ````bash
    ./scripts/libs/install_neuralsat.sh
- nnenum
    ````bash
    ./scripts/libs/install_neuralsat.sh && ./scripts/libs/set_nnenum_mp_count.sh 64

### Run initial verification & filter instances
```bash
source exp_neurips/scripts/run_init_veri.sh
source exp_neurips/scripts/filter_all.sh
```
The logs and results will be in `exp_neurips/results/init_veri`.


### Generate instances
```bash
./exp_neurips/scripts/run_all.sh 0 &
./exp_neurips/scripts/run_all.sh 1 &
./exp_neurips/scripts/run_all.sh 2 &
./exp_neurips/scripts/run_all.sh 3 &
./exp_neurips/scripts/run_all.sh 4 &
```
Generated instances and any logs will be stored in `exp_neurips/generated_instances`.

### Run verification
```bash
source exp_neurips/scripts/run_all.sh
```

The logs will be stored under `exp_neurips/results/veri_logs`, and the final result will will stored in `exp_neurips/results/final_results`.

### Generate figures

```bash
python exp_neurips/scripts/graphing/gen_combined_fig.py
python exp_neurips/scripts/graphing/gen_composition_fig.py
```

This will generate figures used in the paper. The figures will be in `exp_neurips/results/graphs`.



# ReluSplitter

ReluSplitter is a DNN verification(DNNV) benchmark generation tool for ReLU-based network. It takes input as a DNNV instance (network + property) and generates a modified network s.t. the modified network carries the same semantic of the orginal network, but it is harder to verify the property on the modified network. ReluSplitter achieves this by systematically de-stablizing stable neurons in the network. 

![Overview](stuff/figs/tool_overview.PNG)


## Features
Currently, our tool accepts input in the standard `onnx` and `vnnlib` format, and
supports two common layer types.
- Fully-connected layer with ReLU activation
- Convolutional layer with ReLU activation

Technically, any network with above layers can be splitted. However, you might encounter compatibility issues due to factors like ONNX version differences or complex network structures. If you believe your input should work but doesn't, feel free to [open an issue](#) or contact us.


### Why ReluSplitter?

- ✅ **Semantic-preserving**: The generated network always retains the same semantics as the original. For example, if the original instance is `UNSAT`, the modified instance will also be `UNSAT`.  
- ⚡ **Fast generation**: A new DNNV instance can be created within seconds—significantly faster than alternative benchmark generation approaches that require training or distillation.


## Installation
Coming soon

## Usage
basic usage
- info
    ```
    python main.py info --net data/mnist_fc/onnx/mnist-net_256x6.onnx  --spec  data/mnist_fc/vnnlib/prop_5_0.03.vnnlib
    ```
- split
    ```
    python main.py split --net data/mnist_fc/onnx/mnist-net_256x6.onnx  --spec  data/mnist_fc/vnnlib/prop_5_0.03.vnnlib
    ```

ReluSplitter also provide a number of parameters for users to explore. For example: the number of split, the scaling factors, or the type of neuron to split (stably active, stably inactive, unstable). 

## Algorithm and Proff
Summary
### Algorithm
Coming soon

- [Visualized demo of the splitting technique on 1-dimensional input space](https://www.desmos.com/calculator/kthart02fb)

### Proof
Coming soon



