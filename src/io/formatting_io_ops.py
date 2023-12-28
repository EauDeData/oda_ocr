import io
import torch

def preload_model(pytorch_module):
    buffer = io.BytesIO()
    torch.save(pytorch_module, buffer)
    buffer.seek(0)
    return buffer