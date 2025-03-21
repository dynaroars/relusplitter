# ReluSplitter

ReluSplitter is a DNN verification(DNNV) benchmark generation tool for ReLU-based network. It takes input as a DNNV instance (network + property) and generates a modified network s.t. the modified network carries the same semantic of the orginal network, but it is harder to verify the property on the modified network. ReluSplitter achieves this by systematically de-stablizing stable neurons in the network. 

## Features
Currently, our tool accepts input in the standard `onnx` and `vnnlib` format, and
supports two common layer types.
    - Fully-connected layer with ReLU activation
    - Convolutional layer with ReLU activation

Technically, any network with above layers can be splitted. However, it is possible that you will ran into compatibility issues for variouse reasons (e.g. onnx versioning, complex network structures...). If this tool does not work for your and you believe it should, free free to open an issue or contact us.

One advantage of ReluSplitter is that it is *semantic-preserving*, meaning it will always generate a new DNNV instance that has the exact grountruth as the original DNNV instance (e.g., if the orginal DNNV instance is `UNSAT`, then the generated one should also be `UNSAT`). It can *generate a new DNNV instance in seconds*, which is significantly faster than the other benchmark generation techniques, as it does not require training/distillation. 



## Installation
TODO

## Usage
basic usage:
> 123

ReluSplitter also provide a number of parameters for users to explore. For example: the number of split, the scaling factors, or the type of neuron to split (stably active, stably inactive, unstable). 

## Algorithm and Proff
#### Algorithm
Coming soon

#### Proof
Coming soon

# Misc
### [Visualized demo of the splitting technique on 1-dimensional input space](https://www.desmos.com/calculator/kthart02fb)

