import torch

from libs.ImageBind.imagebind.models.imagebind_model import imagebind_huge


def initialize_vlm_model_and_device() -> torch.nn.Module:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_huge(pretrained=True).eval().to(device)
    return model, device
