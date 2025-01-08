import onnxsim
import onnx
import sys


model = onnx.load(sys.argv[1])
model, check = onnxsim.simplify(model)
onnx.save(model, sys.argv[2])