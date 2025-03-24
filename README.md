# ReluSplitter

ReluSplitter is a DNN verification(DNNV) benchmark generation tool for ReLU-based network. It takes input as a DNNV instance (network + property) and generates a modified network s.t. the modified network carries the same semantic of the orginal network, but it is harder to verify the property on the modified network. ReluSplitter achieves this by systematically de-stablizing stable neurons in the network. 

## Features
Currently, our tool accepts input in the standard `onnx` and `vnnlib` format, and
supports two common layer types.
    - Fully-connected layer with ReLU activation
    - Convolutional layer with ReLU activation

Technically, any network with above layers can be splitted. However, you might encounter compatibility issues due to factors like ONNX version differences or complex network structures. If you believe your network should work but doesn't, feel free to [open an issue](#) or contact us directly.


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



