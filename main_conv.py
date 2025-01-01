
import onnx
import torch
from pathlib import Path
from functools import reduce
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.verify import init_verifier
from relu_splitter.utils.read_vnnlib import read_vnnlib


# onnx_path = '/storage/lli/tools/relusplitter/data/collins_rul_cnn/onnx/NN_rul_small_window_20.onnx'
# vnnlib_path = '/home/lli/tools/relusplitter/data/collins_rul_cnn/vnnlib/robustness_2perturbations_delta5_epsilon10_w40.vnnlib'

onnx_path = '/home/lli/tools/relusplitter/data/vggnet16/onnx/vgg16-7.onnx'
vnnlib_path = '/home/lli/tools/relusplitter/data/vggnet16/vnnlib/spec1_football_helmet.vnnlib'

model = onnx.load(onnx_path)
warpped = WarppedOnnxModel(model)
graph = model.graph

input_bound, output_bound = read_vnnlib(vnnlib_path)[0]
input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
model_num_inputs = warpped.num_inputs


# get 1st Conv layer
Conv_1st = warpped.nodes[2]
original_input = warpped.get_node_inputs_no_initializers(Conv_1st)[0]
original_output = Conv_1st.output[0]

print(Conv_1st.op_type)
print(original_input)
print(original_output)
opset_version = model.opset_import
print(f"Current opset version: {opset_version}")
input("continue?")
# for i in range(len(Conv_1st.attribute)):
#     print(Conv_1st.attribute[i])
original_groups = Conv_1st.attribute[0].i
original_dilations = Conv_1st.attribute[1].ints
original_kernel_shape = Conv_1st.attribute[2].ints
original_pads = Conv_1st.attribute[3].ints
original_strides = Conv_1st.attribute[4].ints

if not original_dilations:
    original_dilations = [1,1]

w,b = warpped.get_conv_wb(Conv_1st)
print("Wshape: ", w.shape)
print("Bshape: ", b.shape)
# [OutChannels,InChannels,KernelHeight,KernelWidth]
original_oC, original_iC, original_kH, original_kW = w.shape


# make new W and B
new_w = torch.zeros( (original_oC+1, original_iC, original_kH, original_kW))
new_b = torch.zeros( (original_oC+1,) )

for i in range(original_oC):
    new_w[i] = w[i]
    new_b[i] = b[i]
# new_w[-1], new_w[-2] = w[-1]/2, w[-1]/2
# new_b[-1], new_b[-2] = b[-1]/2, b[-1]/2
temp_w_last = w[-1]
new_w[-1] = temp_w_last / 2
new_w[-2] = temp_w_last / 2
temp_b_last = b[-1]
new_b[-1] = temp_b_last / 2
new_b[-2] = temp_b_last / 2

# assertions
print( torch.all(new_w[-1] == w[-1]/2) )
print( torch.all(new_w[-2] == w[-1]/2) )
print( torch.all(new_b[-1] == b[-1]/2) )
print( torch.all(new_b[-2] == b[-1]/2) )
print( torch.all(new_w[:-2] == w[:-1]) )
print( torch.all(new_b[:-2] == b[:-1]) )
assert torch.allclose(new_w[-1] + new_w[-2], w[-1])
assert torch.allclose(new_b[-1] + new_b[-2], b[-1])


print("New Wshape: ", new_w.shape)
print("New Bshape: ", new_b.shape)

# TODO resume here
# [OutChannels,InChannels,KernelHeight,KernelWidth]
conv1w = torch.zeros( (original_oC, original_oC+1, 1, 1) )
conv1b = torch.zeros( (original_oC) )
for i in range(original_oC):
    conv1w[i,i,0,0] = 1
    conv1b[i] = 0
conv1w[-1,-1,0,0] = 1
conv1w[-1,-2,0,0] = 1
conv1b[-1] = 0

# make initializers
new_w_init = onnx.helper.make_tensor("new_w", onnx.TensorProto.FLOAT, new_w.shape, new_w.flatten())
new_b_init = onnx.helper.make_tensor("new_b", onnx.TensorProto.FLOAT, new_b.shape, new_b.flatten())
conv1w_init = onnx.helper.make_tensor("conv1w", onnx.TensorProto.FLOAT, conv1w.shape, conv1w.flatten())
conv1b_init = onnx.helper.make_tensor("conv1b", onnx.TensorProto.FLOAT, conv1b.shape, conv1b.flatten())

new_node = onnx.helper.make_node(
    "Conv",
    [original_input, "new_w", "new_b"],
    ["Conv_1st_new_lli"],
    name="Conv_1st_new",
    kernel_shape=original_kernel_shape,
    strides=original_strides,
    pads=original_pads,
    dilations=original_dilations,
    # groups=original_groups,
)
new_conv1_node = onnx.helper.make_node(
    "Conv",
    ["Conv_1st_new_lli", "conv1w", "conv1b"],
    [original_output],
    name="Conv_1st_new_lli_conv1x1",
    kernel_shape=[1,1],
    strides=[1,1],
    pads=[0,0,0,0],
    dilations=[1,1],
    # pads=original_pads,
    # strides=original_strides,
    # dilations=original_dilations,
    # groups=original_groups,
)
# print([n.name for n in warpped._model.graph.input])
new_model = warpped.generate_updated_model(
    nodes_to_replace = [Conv_1st],
    additional_nodes = [new_node, new_conv1_node],
    additional_initializers = [new_w_init, new_b_init, conv1w_init, conv1b_init],
    graph_name = "test_conv_split",
    producer_name = "RSplitter_conv"
)
new_model.save(Path("TEST_CONV_SPLIT_1x1.onnx"))
# split one kernel (add one kernel and one bias)
# add a 1x1 Conv layer to merge the additional channel
# connect the new Conv layer to the original Relu layer
# print(warpped.info())

# other checks

print(warpped.input_shapes)
print(new_model.input_shapes)

# vgg0_relu1_fwd
warpped = warpped.truncate_model_at(original_output)
new_model = new_model.truncate_model_at(original_output)
warpped.save(Path("warpped.onnx"))
new_model.save(Path("new_model.onnx"))

# gotta reshape the input to 3 channel input as in vgg16
lb_input = input_lb.reshape(1,3,224,224)
ori_ans = warpped.forward_gpu(lb_input)
new_ans = new_model.forward_gpu(lb_input)
onnx_ori = warpped.forward_onnx(lb_input).to('cuda')
onnx_new = new_model.forward_onnx(lb_input).to('cuda')

# check onnx and gpu output
print("onnx and gpu diff (max)")
print(torch.abs(onnx_ori-ori_ans).max())
print(torch.abs(onnx_new-new_ans).max())
print("onnx and gpu diff (avg)")
print(torch.abs(onnx_ori-ori_ans).mean())
print(torch.abs(onnx_new-new_ans).mean())
print("=======")
print(onnx_new)
print("=======")
print(ori_ans)
print("=======")


print(f"Original model output shape: {ori_ans.shape}")
print(f"New model output shape: {new_ans.shape}")

print("original and new model diff (max)")
print(torch.abs(ori_ans-new_ans).max())

print("original and new model diff (avg)")
print(torch.abs(ori_ans-new_ans).mean())

print("original and new model diff (median)")
print(torch.abs(ori_ans-new_ans).median())

idx_max = torch.argmax(torch.abs(ori_ans - new_ans))
print(f"Max difference location: {idx_max}")
print(f"Original value: {ori_ans.flatten()[idx_max]}")
print(f"New value: {new_ans.flatten()[idx_max]}")
