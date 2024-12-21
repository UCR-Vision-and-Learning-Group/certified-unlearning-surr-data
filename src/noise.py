from torch.nn.functional import softmax, one_hot
import torch


from src.utils import get_module_device


def approximate_upper_cross_entropy(loader, model):
    device = get_module_device(model)
    required_probs = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        probs = softmax(output, dim=1)
        mask = one_hot(label.long(), num_classes=probs.size(0)).bool()
        required_probs.append(probs[mask])
    return torch.sum(torch.log(torch.cat(required_probs))) / len(loader.dataset)


def approximate_entropy():
    pass


def set_noise(loader, eps, delta, beta=1, alpha=1):
    pass