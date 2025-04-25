# ReluSplitter - VNNCOMP-25

## Benchmark Generation

To generate benchmark instances:

1. Run the installation script:
   ```bash
   ./scripts/install.sh
   ```

2. Generate properties:

    ```bash
    python generate_properties.py
    ```

This benchmark uses selected instances from previous VNNCOMP iterations as seed inputs to generate more challenging instances. The networks include both fully connected (FC) and convolutional (Conv) architectures. All networks uses standard feed-forward structure, except for SRI ResNet A/B, which includes residual connections.

<small>
We are considering splitting this benchmark into two smaller benchmarks: one will only have FC networks, and the other will include only Conv networks, which may have residual connections. We think this could encourage more participation, as some tools may not work well with or may struggle to handle convolutional networks. If necessary, we can also exclude certain benchmarks from the seed instances.
</small>


## Seed Benchmarks Used:
The following benchmarks from previous VNNCOMP iterations were used:

- ACAS Xu (https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/acasxu)
    The networks in this benchmark was simplified (we merged the Matmul and Add nodes to GEMM nodes to make it compatible with higher onnx version)
- mnist_fc (https://github.com/pat676/mnist_fc_vnncomp2022.git)
- Oval21 (https://github.com/alessandrodepalma/oval21-benchmark.git)
- SRI ResNet A/B (https://github.com/mnmueller/vnn_comp_22_bench_sri.git)
- Cifar Biasfield (https://github.com/pat676/cifar_biasfield_vnncomp2022.git)























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
