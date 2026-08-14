import torch
import transformers
import accelerate
import psutil
import pynvml

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0))
print("transformers=", transformers.__version__)
print("accelerate=", accelerate.__version__)
print("environment=PASS")