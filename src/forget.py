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
import math
import argparse
from torch.nn.functional import softmax, one_hot

# from src.noise import set_noise
from src.utils import get_module_device
from remedi.train_complete import train_complete_model


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


def calculate_grad_norm(grad):
    grad, prev_sizes = _linearize_grads(grad)
    return torch.norm(grad).item(), prev_sizes


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


######################################### NOISE
def approximate_upper_cross_entropy(loader, model):
    device = get_module_device(model)
    required_probs = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        probs = softmax(output, dim=1)
        mask = one_hot(label.long(), num_classes=probs.size(1)).bool()
        required_probs.append(probs[mask])
    return -1 * (torch.sum(torch.log(torch.cat(required_probs))).item() / len(loader.dataset))


def approximate_entropy(train_dataset, val_dataset, device):
    args_data = argparse.Namespace()
    args_data.type = 'custom'
    if isinstance(train_dataset[0], tuple) or isinstance(train_dataset[0], list):
        args_data.dim = len(train_dataset[0][0])
    else:
        args_data.dim = len(train_dataset[0])

    args_KNIFE = argparse.Namespace()
    args_KNIFE.batchsize = 128
    args_KNIFE.num_modes = 128
    args_KNIFE.epochs = 30
    args_KNIFE.lr = 1e-3
    args_KNIFE.shuffle = True
    args_KNIFE.cov_diagonal = 'var'
    args_KNIFE.cov_off_diagonal = 'var'
    args_KNIFE.average = 'var'
    args_KNIFE.use_tanh = True
    args_KNIFE.device = device
    args_KNIFE.dimension = args_data.dim
    args_KNIFE.custom = True

    knife, train_losses, val_losses = train_complete_model(args_data=args_data, args_REMEDI=None, args_KNIFE=args_KNIFE,
                                                           device=device, custom_trd=train_dataset,
                                                           custom_td=val_dataset, knife_only=True)

    return train_losses[-1]


def calculate_upper_tv(surr_loader, model,
                       tighten_bound, train_dataset, val_dataset):
    device = get_module_device(model)

    app_up_cross = approximate_upper_cross_entropy(surr_loader, model)
    if tighten_bound:
        app_ent = approximate_entropy(train_dataset, val_dataset, device)
    else:
        app_ent = 0

    upper_kl = app_up_cross - app_ent
    return math.sqrt(2 * upper_kl)


def calculate_upper_app_unlearn_surr(surr_loader, model, grad, smooth, sc,
                                     tighten_bound, train_dataset, val_dataset, surr=True):
    grad_norm, prev_sizes = calculate_grad_norm(grad)
    if surr:
        upper = smooth / (sc ** 2)
        upper_tv = calculate_upper_tv(surr_loader, model, tighten_bound, train_dataset, val_dataset)
        upper = upper * upper_tv * grad_norm
        return upper, prev_sizes
    else:
        return None, prev_sizes


def calculate_upper_retrain_app_unlearn(num_retain, num_forget, sc, lip, hlip):
    numerator = 2 * hlip * (lip ** 2) * (num_forget ** 2)
    denominator = (sc ** 3) * (num_retain ** 2)
    return numerator / denominator


def set_noise(surr_loader, forget_size, model, grad,
              eps, delta, smooth=1, sc=1, lip=1, hlip=1,
              tighten_bound=True, train_dataset=None, val_dataset=None, surr=False):
    upper_retrain_app_unlearn = calculate_upper_retrain_app_unlearn(len(surr_loader.dataset),
                                                                    forget_size,
                                                                    sc, lip, hlip)

    upper_app_unlearn_surr, prev_sizes = calculate_upper_app_unlearn_surr(surr_loader, model, grad,
                                                                          smooth, sc,
                                                                          tighten_bound, train_dataset, val_dataset,
                                                                          surr=surr)
    if surr:
        upper = upper_retrain_app_unlearn + upper_app_unlearn_surr
    else:
        upper = upper_retrain_app_unlearn
    sigma = upper / eps
    sigma = sigma * math.sqrt(2 * math.log(1.25 / delta))

    model_size = 0
    for size in prev_sizes:
        model_size += size[0] * size[1]

    update = torch.normal(0, sigma, size=(model_size,))
    update = _adjust_update(update, prev_sizes)
    return update


#########################################


def forget(model, whess_loader, fhess_loader, grad_loader, criterion, linear=False, num_class=None,
           eps=None, delta=None, smooth=1, sc=1, lip=1, hlip=1,
           tighten_bound=True, train_dataset=None, val_dataset=None, surr=False):
    device = get_module_device(model)
    fmodel = deepcopy(model.to('cpu')).to(device)
    hess = calculate_hess(fmodel, whess_loader, fhess_loader, criterion, linear=linear, num_class=num_class)
    grads = calculate_grad(fmodel, grad_loader, criterion)
    update = calculate_update(hess, grads, device)
    update_model(fmodel, update)
    if eps is not None and delta is not None:
        noise = set_noise(whess_loader, len(fhess_loader.dataset), model, grads, eps, delta,
                          smooth=smooth, sc=sc, lip=lip, hlip=hlip,
                          tighten_bound=tighten_bound, train_dataset=train_dataset, val_dataset=val_dataset,
                          surr=surr)
        update_model(fmodel, noise)
    return fmodel
