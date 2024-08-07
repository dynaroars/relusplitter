import torch
import onnx
import numpy as np
from onnx2pytorch import ConvertModel
from onnxruntime import InferenceSession

from random import uniform


from pathlib import Path
import uuid

def get_random_id(len=8):
    assert len <= 32
    return str(uuid.uuid4())[:len]



def generate_random_inputs(input_shape, num_tests):
    inputs = []
    for _ in range(num_tests):
        input_data = np.random.uniform(-100, 100, input_shape)
        inputs.append(input_data.astype(np.float32))
    return inputs

def compare_outputs(output1, output2, tolerance=1e-5):
    return np.allclose(output1, output2, atol=tolerance)

def check_model_equivalency(m1, m2, input_shape, n=100, **kwargs):
    for _ in range(n):
        x = torch.randn(input_shape)
        y1,y2 = m1.forward(x), m2.forward(x)
        if not torch.allclose(y1, y2, **kwargs):
            cex.append((x,y1,y2))
            logger.error(f"Outputs are not the same for input {x}\n{y1}\n{y2}")
            return False
    return True



def check_model_equivalency(m1, m2, input_shape, num_tests=100):
    # Check type equivalency
    # randomly sample num_tests inputs and compare outputs
    # doesnt provide any formal guarantees, but should be good enough for most cases

    assert type(m1) == type(m2) or (isinstance(m1, torch.nn.Module) and isinstance(m1, torch.nn.Module)) 
    
    # Generate random inputs
    random_inputs = generate_random_inputs(input_shape, num_tests)

    if isinstance(m1, torch.nn.Module):
        m1.eval()
        m2.eval()
        
        for input_data in random_inputs:
            input_tensor = torch.tensor(input_data)
            with torch.no_grad():
                output1 = m1(input_tensor).numpy()
                output2 = m2(input_tensor).numpy()
            assert compare_outputs(output1, output2), "Outputs are not the same for PyTorch models"
    
    elif isinstance(m1, ConvertModel):
        for input_data in random_inputs:
            output1 = m1.forward(*input_data)
            output2 = m2.forward(*input_data)
            assert compare_outputs(output1, output2), "Outputs are not the same for ConvertModel"
    
    elif isinstance(m1, onnx.ModelProto):
        s1 = InferenceSession(m1.SerializeToString())
        s2 = InferenceSession(m2.SerializeToString())
        
        for input_data in random_inputs:
            input_name = s1.get_inputs()[0].name
            output1 = s1.run(None, {input_name: input_data})[0]
            output2 = s2.run(None, {input_name: input_data})[0]
            assert compare_outputs(output1, output2), "Outputs are not the same for ONNX models"
    
    elif isinstance(m1, Path):
        if m1.suffix == '.onnx':
            s1 = InferenceSession(m1.as_posix())
            s2 = InferenceSession(m2.as_posix())
            
            for input_data in random_inputs:
                input_name = s1.get_inputs()[0].name
                output1 = s1.run(None, {input_name: input_data})[0]
                output2 = s2.run(None, {input_name: input_data})[0]
                assert compare_outputs(output1, output2), "Outputs are not the same for ONNX models from file"
        
        elif m1.suffix == '.pth':
            model1 = torch.load(m1)
            model2 = torch.load(m2)
            model1.eval()
            model2.eval()
            
            for input_data in random_inputs:
                input_tensor = torch.tensor(input_data)
                with torch.no_grad():
                    output1 = model1(input_tensor).numpy()
                    output2 = model2(input_tensor).numpy()
                assert compare_outputs(output1, output2), "Outputs are not the same for PyTorch models from file"
        else:
            raise ValueError(f"Unsupported file type: {m1.suffix}")
    else:
        raise ValueError(f"Unsupported type: {type(m1)}")

    print("All tests passed. The models are equivalent.")




