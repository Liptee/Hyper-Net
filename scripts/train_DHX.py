import torch
import torch.nn as nn

def get_model(name:str, cuda_dev, **kwargs):
    device = kwargs.setdefault("device", cuda_dev)
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    kwargs.setdefault("patch_size", 1)
    center_pixel = True
    model = HuEtal(n_bands, n_classes)

class HuEtal(nn.Module):
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)