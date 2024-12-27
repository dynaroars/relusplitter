from .common import *

class RSplitter_conv():

    def check_conv():
        # check if is 2d conv

    def split_conv():
        # input bound -> output bounds
        # Some notes:
        # num kernels = num output channels = c_out
        # num input channels = c_in
        # kernel size = kh x kw

        # prioritize splitting kernel that have less unstable bounds
        # each kernel will produce oH x oW output bounds (worry about computation later)
        
        # step 1. get the output boudns and kinda put them in pairs (kernel, output bounds)

        # step 2. idk like sort the kernels by the number of stable/unstable bounds

        # step 3. split the kernels
        # into two kernel with a merge layer, like with the FC layer
        # OR
        # Can use 1x1 conv to merge the additional channel


        c_in, c_out = None, None
        kh, kw = None, None


    def get_baseline_split_conv():
        pass