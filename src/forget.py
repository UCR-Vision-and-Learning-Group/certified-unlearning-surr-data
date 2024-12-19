# TODO: here is all experimental calculating the hessian of the whole network seems like non-trivial
#  what I can do is first I will go with the simple linear classifiers to calculate the Hessian really fast,
#  then I will look for implementations to calculate the Hessian of the network

# import torch
# import torch.autograd.functional as autofunc
#
# def _set_model_params(model, flat_params):
#     idx = 0
#     for param in model.parameters():
#         num = param.numel()
#         param.data.copy_(flat_params[idx:idx + num].view_as(param))
#         idx += num
#
#
# # Function to flatten model parameters
# def _get_flat_params(model):
#     return torch.cat([p.view(-1) for p in model.parameters()])
#
#
# def _loss_wrapper(model, criterion, data, label):
#     preds = model(data)
#     loss = criterion(preds, label)
#     return loss
#
#
# def compute_batch_grad_hess(model, criterion, data, label, flat_params):
#     _set_model_params(model, flat_params)
#     loss = _loss_wrapper(model, criterion, data, label)
#     grad = torch.autograd.grad(loss, flat_params, create_graph=True)[0]
#     hess = autofunc.hessian(lambda params: _loss_wrapper(model, criterion, data, label), flat_params)
#     return grad, hess
#
#
# def accumulate_grad_hess(model, criterion, loader):
#     # Flatten model parameters
#     flat_params = _get_flat_params(model).clone().requires_grad_(True)
#
#     # Initialize accumulators for gradients and Hessians
#     grad = torch.zeros_like(flat_params)
#     hess = torch.zeros(flat_params.size(0), flat_params.size(0))
#
#     for (data, label) in loader:
#         curr_grad, curr_hess = compute_batch_grad_hess(model, criterion, data, label, flat_params)

import torch
from copy import deepcopy

from src.utils import get_module_device


def calculate_cov(loader, device):
    cumulative_cov = None
    cumulative_size = 0
    for data, _ in loader:
        data = data.to(device)
        if cumulative_cov is None:
            cumulative_cov = (data.T @ data).clone().detach().to('cpu')
        else:
            cumulative_cov += (data.T @ data).clone().detach().to('cpu')
        cumulative_size = cumulative_size + data.shape[0]
    return cumulative_cov / cumulative_size


def calculate_hess(model, wloader, floader, criterion, linear=False, num_class=None):
    device = get_module_device(model)
    if linear and num_class is not None:
        whess = torch.kron(torch.eye(num_class), calculate_cov(wloader, device))
        fhess = torch.kron(torch.eye(num_class), calculate_cov(floader, device))
        wsize = len(wloader.dataset)
        fsize = len(floader.dataset)
        return (wsize * whess - fsize * fhess) / (wsize - fsize)


def _accumulate_grads(prev, curr, prev_size, curr_size):
    if prev is None:
        return curr
    else:
        res = []
        for idx in range(len(prev)):
            res.append((prev[idx] * prev_size + curr[idx] * curr_size) / (prev_size + curr_size))
        return res


def calculate_grad(model, loader, criterion):
    device = get_module_device(model)
    grads = None
    total_size = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        curr_grads = [param.grad.clone().detach().to('cpu') for param in model.parameters()]
        grads = _accumulate_grads(grads, curr_grads, total_size, total_size + data.shape[0])
        total_size += data.shape[0]
    return grads


def _linearize_grads(grads):
    flat_grad = [grad.view(-1) for grad in grads]
    prev_sizes = [grad.size() for grad in grads]
    return torch.cat(flat_grad), prev_sizes


def _adjust_update(update, prev_sizes):
    splits = [size[0] * size[1] for size in prev_sizes]
    adjusted_update = torch.split(update, splits)
    adjusted_update = [adjusted_update[idx].view(prev_sizes[idx]) for idx in range(len(prev_sizes))]
    return adjusted_update


def calculate_update(hessian, grads, device):
    flat_grad, prev_sizes = _linearize_grads(grads)
    flat_grad = flat_grad.to(device)
    eps = 1e-4
    hess = (hessian + eps * torch.eye(hessian.shape[0])).to(device)
    inv = torch.linalg.inv(hess)
    update = torch.mv(inv, flat_grad).to('cpu')
    update = _adjust_update(update, prev_sizes)
    return update


def update_model(model, updates):
    params = [param for param in model.parameters()]
    device = get_module_device(model)
    with torch.no_grad():
        for param, update in zip(params, updates):
            param.add_(update.to(device))


def forget(model, whess_loader, fhess_loader, grad_loader, criterion, linear=False, num_class=None):
    device = get_module_device(model)
    fmodel = deepcopy(model.to('cpu')).to(device)
    hess = calculate_hess(fmodel, whess_loader, fhess_loader, criterion, linear=linear, num_class=num_class)
    grads = calculate_grad(fmodel, grad_loader, criterion)
    update = calculate_update(hess, grads, device)
    update_model(fmodel, update)
    return fmodel
