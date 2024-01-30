import io
import torch

def preload_model(pytorch_module):
    print('trying to preload linear models?')
    buffer = io.BytesIO()
    torch.save(pytorch_module, buffer)
    buffer.seek(0)
    print('i preloaded it')
    return buffer

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
