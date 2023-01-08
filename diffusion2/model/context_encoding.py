import torch

def lower_resolution(x):
    context = torch.nn.functional.interpolate(x, scale_factor=0.5)
    context = torch.nn.functional.interpolate(context, scale_factor=2)
    return x

class ContextEncoding():
    def __init__(self, code_name):
        self.code_name = code_name
    
    def __call__(self, x):
        if self.code_name == "lower_resolution":
            return lower_resolution(x)

        raise NotImplementedError