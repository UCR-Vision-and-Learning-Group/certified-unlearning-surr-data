import torch

from tqdm import tqdm
from src.loss import L2RegularizedCrossEntropyLoss


def evaluate(test_loader, model, criterion, device=None, log=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    pbar = tqdm(test_loader, desc='eval', unit='batch')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(criterion, L2RegularizedCrossEntropyLoss):
                loss = criterion(outputs, targets, model)
            else:
                loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=correct / total)
        pbar.close()
    if log:
        return correct / total
