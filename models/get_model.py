from .resnet import resnet18
from .wideresnet import WideResNet28_10

def get_model(model_name, device):
    if model_name == "ResNet18":
        net = resnet18()
    elif model_name == "WideResNet-28-10":
        net = WideResNet28_10()
    else:
        raise ValueError(f"Invalid model name '{model_name}'. Available options are: 'ResNet18', 'WideResNet-28-10'.")
    net = net.to(device)
    return net
