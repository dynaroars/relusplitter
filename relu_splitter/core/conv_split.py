from .common import *

class RSplitter_conv():

    def check_conv():
        pass
        # check if is 2d conv

    def get_conv_patch():
        # get the idxs of the patch responsible for the output
        pass

    def compute_split_layer_bias(self, nodes_to_split, kernel_idx, mode="max_unstablize"):
        # find the appropriate bias for the given kernel in the split layer
        # a. Compute the output bound of the Conv layer (pre-relu)
        # b. find the value that satisfied the most patches
        # c. return the value
        # ONE PROBLEM: what if the split NVM
        conv_node, relu_node = nodes_to_split
        output_lb,output_ub = self.warpped_model.get_bound_of(self.bounded_input, conv_node.output[0], method="ibp")
        self.logger.debug(f"bound shapes: {output_lb.shape}, {output_ub.shape}")
        # TODO: filter bounds based on neurons stability
        # keep only stable neurons lb & ub have same sign (use torch)
        output_lb, output_ub = output_lb.flatten(), output_ub.flatten()
        stable_idxs = torch.where((output_lb * output_ub) > 0)
        lb, ub = output_lb[stable_idxs], output_ub[stable_idxs]

        # find the val to sat most intervals
        evnts = []
        for i in range(len(lb)):
            evnts.append((lb[i].item(), 1))
            evnts.append((ub[i].item(), -1))
        evnts.sort(key=lambda x: x[0])
        active_intervals = 0
        max_active_intervals = -1
        start, end = None, None
        curr_start = evnts[0][0]
        for val, evnt in evnts:
            active_intervals += evnt
            if active_intervals > max_active_intervals:
                max_active_intervals = active_intervals
                start = curr_start
                end = val
            curr_start = val

        bias = random.uniform(start, end)  
        self.logger.info(f"Max active intervals: {max_active_intervals}/{len(lb)}/{len(output_lb)}")
        self.logger.info(f"Max active interval range: [{start}, {end}]")
        self.logger.info(f"Selected bias: {bias}")
        # return a random value in the range using random
        return bias



    def split_conv_node(self, nodes_to_split,kernel_idxs, split_locs, scale_factors=(1.0, -1.0), manual_check_io=False):
        conv_node, relu_node = nodes_to_split
        assert conv_node.op_type == "Conv" and relu_node.op_type == "Relu"
        assert len(kernel_idxs) == len(split_locs)

        ori_model = self.warpped_model
        ori_graph_name = ori_model.graph_name

        # if manual_check_io:
        ori_input = conv_node.input[0]
        ori_output = relu_node.output[0]

        ori_groups = conv_node.attribute[0].i
        ori_dilations = conv_node.attribute[1].ints
        ori_kernel_shape = conv_node.attribute[2].ints
        ori_pads = conv_node.attribute[3].ints
        ori_strides = conv_node.attribute[4].ints

        ori_w, ori_b = ori_model.get_conv_wb(conv_node)
        assert ori_w.dim() == 4
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape



        # make weights and bias for split and merge layers
        num_splits = len(kernel_idxs)
        new_oC = ori_oC + num_splits

        new_w = torch.zeros( (new_oC, ori_iC, ori_kH, ori_kW))
        new_b = torch.zeros( (new_oC,) )
        merge_w = torch.zeros( (ori_oC, new_oC, 1, 1) )
        merge_b = torch.zeros( (ori_oC,) )

        curr_idx = 0
        scale_ratio_pos, scale_ratio_neg = scale_factors
        for i in tqdm(range(ori_oC), desc="Constructing new Conv layer..."):
            split_loc = split_locs[i]
            temp_w = ori_w[i]
            # temp_b = ori_w[i] @ split_loc
            temp_b = self.compute_split_layer_bias(nodes_to_split, i)
            # for FC the temp_b is one single value, but for conv this is a vector/matrix (the whole output channel). 
            # Basically each patch has a corresponding temp_b value.
            # temp_b is the displacement from the origin (0)
            # so we use the vector to infer a best temp_b vlaue that can unstablize most patches.
            # 1. during the split point calc, try to minimize the range of temp_b
            # 2. during the split point calc, instead of using concret split point, use a range.
            #    this way, in the later calculation of temp_b, we will have a set of ranges, and 
            #    we can choose the one that can unstablize most patches.
            #    If the selected patch x kernel is stable on the split range, we can split at any point in the range.
            #    and any temp_b in the range can make the splitted kernel x patch unstable.
            if i in kernel_idxs:
                # split the kernel
                new_w[curr_idx]     = temp_w * scale_ratio_pos
                new_w[curr_idx+1]   = temp_w * scale_ratio_neg
                new_b[curr_idx]     = temp_b * -scale_ratio_pos
                new_b[curr_idx+1]   = temp_b * -scale_ratio_neg
                # set merge w & b
                merge_w[i, curr_idx  , 0, 0] = 1/scale_ratio_pos
                merge_w[i, curr_idx+1, 0, 0] = 1/scale_ratio_neg
                merge_b[i] = ori_b[i] + temp_b

                curr_idx += 2
            else:
                # copy the kernel
                new_w[curr_idx] = ori_w[i]
                new_b[curr_idx] = ori_b[i]
                # forward through merge layer
                merge_w[i, curr_idx, 0, 0] = 1
                merge_b[i] = 0
                curr_idx += 1
        # compose the new onnx nodes
        # tensor names
        input_tensor_name       = ori_input
        output_tensor_name      = ori_output
        split_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPre")
        split_postRelu_tensor_name  = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPost")
        merge_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mPre")
        merge_postRelu_tensor_name  = output_tensor_name
        # initializer names
        split_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sW")
        split_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sB")
        merge_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mW")
        merge_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mB")
        # node names
        split_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_s")
        split_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_sReLU")
        merge_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_m")
        merge_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_mReLU")

        split_conv_layer = onnx.helper.make_node(
            'Conv',
            inputs=[input_tensor_name, split_w_init_name, split_b_init_name],
            outputs=[split_preRelu_tensor_name],
            name=split_node_name,
            # group=ori_groups,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        split_relu_layer = onnx.helper.make_node(
            'Relu',
            inputs=[split_preRelu_tensor_name],
            outputs=[split_postRelu_tensor_name],
            name=split_relu_node_name
        )
        merge_conv_layer = onnx.helper.make_node(
            'Conv',
            inputs=[split_postRelu_tensor_name, merge_w_init_name, merge_b_init_name],
            outputs=[merge_preRelu_tensor_name],
            name=merge_node_name,
            # group=ori_groups,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )
        merge_relu_layer = onnx.helper.make_node(
            'Relu',
            inputs=[merge_preRelu_tensor_name],
            outputs=[merge_postRelu_tensor_name],
            name=merge_relu_node_name
        )

        new_nodes = [split_conv_layer, split_relu_layer, merge_conv_layer, merge_relu_layer]
        new_initializers = [
            onnx.helper.make_tensor(split_w_init_name, onnx.TensorProto.FLOAT, new_w.shape, new_w.flatten()),
            onnx.helper.make_tensor(split_b_init_name, onnx.TensorProto.FLOAT, new_b.shape, new_b.flatten()),
            onnx.helper.make_tensor(merge_w_init_name, onnx.TensorProto.FLOAT, merge_w.shape, merge_w.flatten()),
            onnx.helper.make_tensor(merge_b_init_name, onnx.TensorProto.FLOAT, merge_b.shape, merge_b.flatten())
        ]

        new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[conv_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=ori_graph_name,
                                                                producer_name="ReluSplitter")
        return new_model


        



    def get_baseline_split_conv():
        pass