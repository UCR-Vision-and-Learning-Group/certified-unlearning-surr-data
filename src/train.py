import torch

from tqdm import tqdm

from src.eval import evaluate


def train_epoch(train_loader, model, criterion, optimizer, epoch, device):
    pbar = tqdm(train_loader, desc='train epoch {}'.format(epoch), unit='batch')
    for data, target in tqdm(train_loader, desc="train"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        pbar.update(1)


def train(train_loader, val_loader, model, criterion, optimizer, num_epoch=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    for epoch in range(num_epoch):
        train_epoch(train_loader, model, criterion, optimizer, epoch=epoch, device=device)
        evaluate(val_loader, model, criterion, device=device)
