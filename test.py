# create a test onnx model with a specified activation function
# input -> gemm -> activation -> output

# activation can be ReLU, LeakyReLU, PReLU, thresholdedReLU

import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort


input_size = 4
output_size = 3
batch_size = 1



def gen_onnx_model(activation_type="ReLU"):
    assert activation_type in ["ReLU", "LeakyReLU", "PReLU", "ThresholdedRelu"], "Unsupported activation type"

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, input_size])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, output_size])
    weight_tensor = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[output_size, input_size],
        vals=np.random.randn(output_size, input_size).astype(np.float32).flatten().tolist()
    )
    bias_tensor = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[output_size],
        vals=np.random.randn(output_size).astype(np.float32).flatten().tolist()
    )
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["input", "weight", "bias"],
        outputs=["gemm_output"],
        transB=1
    )

    if activation_type == "ReLU":
        activation_node = helper.make_node(
            "Relu",
            inputs=["gemm_output"],
            outputs=["output"]
        )
    elif activation_type == "LeakyReLU":
        activation_node = helper.make_node(
            "LeakyRelu",
            inputs=["gemm_output"],
            outputs=["output"],
            alpha=0.01
        )
    elif activation_type == "PReLU":
        slope_tensor = helper.make_tensor(
            name="slope",
            data_type=TensorProto.FLOAT,
            dims=[output_size],
            vals=np.random.randn(output_size).astype(np.float32).flatten().tolist()
        )
        activation_node = helper.make_node(
            "PRelu",
            inputs=["gemm_output", "slope"],
            outputs=["output"]
        )
    elif activation_type == "ThresholdedRelu":
        activation_node = helper.make_node(
            "ThresholdedRelu",
            inputs=["gemm_output"],
            outputs=["output"],
            alpha=1.0
        )
    graph = helper.make_graph(
        nodes=[gemm_node, activation_node],
        name="FC_Activation_Graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor, bias_tensor] + ([slope_tensor] if activation_type == "PReLU" else [])
    )
    model = helper.make_model(graph)
    return model


def spliter_transformation_LeakeyReLU(model, gemm_node, activation_node):
    assert activation_node.op_type == "LeakyRelu", "The activation function is not LeakyReLU"

    # randomly pick half of the neurons to split, keep idxs
    split_idxs = np.random.choice(range(output_size), size=output_size // 2, replace=False).tolist()
    
    



if __name__ == "__main__":
    activation_types = ["ReLU", "LeakyReLU", "PReLU", "ThresholdedRelu"]
    for act_type in activation_types:
        model = gen_onnx_model(activation_type=act_type)
        onnx.save(model, f"fc_{act_type.lower()}.onnx")
        print(f"Saved model with {act_type} activation as fc_{act_type.lower()}.onnx")


# test the generated models
import onnx
import onnxruntime as ort
import numpy as np  
activation_types = ["ReLU", "LeakyReLU", "PReLU", "ThresholdedRelu"]
for act_type in activation_types:
    model_path = f"fc_{act_type.lower()}.onnx"
    model = onnx.load(model_path)
    # print the model graph
    print(f"Model graph for {act_type} activation:")
    print(onnx.helper.printable_graph(model.graph))
    for node in model.graph.node:
        print(f"Node: {node.name}, OpType: {node.op_type}")
    # Create inference session
    ort_session = ort.InferenceSession(model_path)

    # Generate random input
    input_data = np.random.randn(1, 4).astype(np.float32)

    # Run inference
    outputs = ort_session.run(None, {"input": input_data})
    print(f"Output for model with {act_type} activation:")
    print(outputs[0])