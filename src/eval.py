import torch

from tqdm import tqdm


def evaluate(test_loader, model, criterion, device=None):
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
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=loss.item(), acc=correct / total)
        pbar.close()
