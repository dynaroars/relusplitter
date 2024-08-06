import onnx
import onnxsim
from onnxsim import simplify

from pathlib import Path
import os


def simplify_acasxu():
    acasxu_onnx_root = Path("data/acasxu/onnx")
    output_root = Path("data/acasxu_converted/onnx")

    output_root.mkdir(parents=True, exist_ok=True)

    for onnx_file in acasxu_onnx_root.rglob("*.onnx"):
        print(f"Simplifying {onnx_file}")        
        dest     = output_root / (onnx_file.stem + "_converted.onnx")
        basename = os.path.basename(onnx_file)

        model = onnx.load(onnx_file)
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"

        # Remove the initializer from the input
        model_input  = model_simp.graph.input
        model_output = model_simp.graph.output
        initializer_names = [i.name for i in model_simp.graph.initializer]

        new_input = [i for i in model_input if i.name not in initializer_names]
        new_output = model_output

        assert len(new_input) == 1, "There should be only one input"
        assert len(new_output) == 1, "There should be only one output"

        new_graph = onnx.helper.make_graph(
            model_simp.graph.node,
            f"{basename}_simplified",
            new_input,
            new_output,
            model_simp.graph.initializer
        )
        new_model = onnx.helper.make_model(new_graph, producer_name='Relu-splitter-EXP',)

        # check conversion correctness

        onnx.save(new_model, dest)

if __name__ == "__main__":
    simplify_acasxu()