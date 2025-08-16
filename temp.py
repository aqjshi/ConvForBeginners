import torch
import torch.nn as nn
import numpy as np
import time
from openvino.runtime import Core


'''
bathc sizen 100000
Model exported to ONNX
CPU result: [[2. 4. 6.]]
GPU result: [[2. 4. 6.]]
CPU time for 1000 inferences: 3.1534 seconds
GPU time for 1000 inferences: 0.2226 seconds



'''



# Step 1: Export PyTorch model to ONNX
class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
dummy_input = torch.randn(1, 3)
torch.onnx.export(model, dummy_input, "simple_model.onnx", input_names=["input"], output_names=["output"])
print("Model exported to ONNX")

# Step 2: Load ONNX model with OpenVINO
core = Core()
model_onnx = core.read_model("simple_model.onnx")

# Step 3: Compile for CPU
compiled_cpu = core.compile_model(model_onnx, "CPU")
infer_cpu = compiled_cpu.create_infer_request()

# Step 4: Compile for GPU (Intel)
compiled_gpu = core.compile_model(model_onnx, "GPU")
infer_gpu = compiled_gpu.create_infer_request()

# Step 5: Prepare input
input_tensor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

# Step 6: Benchmark CPU
start_cpu = time.time()
for _ in range(100000):
    result_cpu = infer_cpu.infer({compiled_cpu.input(0): input_tensor})
end_cpu = time.time()

# Step 7: Benchmark GPU
start_gpu = time.time()
for _ in range(1000):
    result_gpu = infer_gpu.infer({compiled_gpu.input(0): input_tensor})
end_gpu = time.time()

# Step 8: Print results
print("CPU result:", result_cpu[compiled_cpu.output(0)])
print("GPU result:", result_gpu[compiled_gpu.output(0)])
print(f"CPU time for 1000 inferences: {end_cpu - start_cpu:.4f} seconds")
print(f"GPU time for 1000 inferences: {end_gpu - start_gpu:.4f} seconds")
