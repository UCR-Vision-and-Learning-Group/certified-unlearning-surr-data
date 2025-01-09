import torch
import torch.nn as nn

from tqdm import tqdm

from src.eval import evaluate
from src.loss import L2RegularizedCrossEntropyLoss
from archive.vae import loss_function
import math
from src.utils import get_module_device


def train_epoch(train_loader, model, criterion, optimizer, epoch, device):
    pbar = tqdm(train_loader, desc='train epoch {}'.format(epoch + 1), unit='batch')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if isinstance(criterion, L2RegularizedCrossEntropyLoss):
            loss = criterion(output, target, model)
        else:
            loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    pbar.close()


def train(train_loader, val_loader, model, criterion, optimizer, num_epoch=10, device=None, target_acc=None,
          threshold=0.05):
    device = get_module_device(model)
    model.train()
    for epoch in range(num_epoch):
        train_epoch(train_loader, model, criterion, optimizer, epoch=epoch, device=device)
        if target_acc is None:
            evaluate(val_loader, model, criterion, device=device)
        else:
            acc = evaluate(val_loader, model, criterion, device=device, log=True)
            if abs(target_acc - acc) < threshold:
                return (epoch + 1) * len(train_loader.dataset)


def train_vae(train_loader, vae, optimizer, num_epoch=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.train()
    for epoch in range(num_epoch):
        pbar = tqdm(train_loader, desc='vae train epoch {}'.format(epoch + 1), unit='batch')
        for batch in pbar:
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch, _ = batch
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, batch, mu, logvar = vae(batch)
            loss = loss_function(recon, batch, mu, logvar)
            loss['loss'].backward()
            optimizer.step()
            pbar.set_postfix(loss=loss['loss'].item(), recon=loss['Reconstruction_Loss'].item(), kld=loss['KLD'].item())
        pbar.close()

