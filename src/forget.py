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
from torch.nn.functional import softmax, one_hot
from tqdm import tqdm
import os
import logging
from torch.utils.data import DataLoader, TensorDataset

from src.utils import get_module_device, freeze_model, melt_model
from src.loss import L2RegularizedCrossEntropyLoss
from src.dv import train_dv_bound


def batched_kronecker_product(A, B):
    """
    Compute the batched Kronecker product of two tensors A and B.

    Parameters:
        A: Tensor of shape (batch_size, m, n)
        B: Tensor of shape (batch_size, p, q)

    Returns:
        Tensor of shape (batch_size, m*p, n*q) representing the batched Kronecker product.
    """
    # Get the shapes
    batch_size, m, n = A.shape
    _, p, q = B.shape

    # Reshape tensors to enable broadcasting
    A_expanded = A.unsqueeze(-1).unsqueeze(-3)  # Shape: (batch_size, m, 1, n, 1)
    B_expanded = B.unsqueeze(-2).unsqueeze(-4)  # Shape: (batch_size, 1, p, 1, q)

    # Elementwise multiplication
    kron = A_expanded * B_expanded  # Shape: (batch_size, m, p, n, q)

    # Reshape to final shape (batch_size, m*p, n*q)
    kron = kron.reshape(batch_size, m * p, n * q)

    return torch.sum(kron, dim=0)


def batched_outer(A, B):
    """
    Compute the batched outer product of two tensors A and B.

    Parameters:
        A: Tensor of shape (batch_size, m)
        B: Tensor of shape (batch_size, n)

    Returns:
        Tensor of shape (batch_size, m, n)
    """
    # Reshape A to (batch_size, m, 1) and B to (batch_size, 1, n)
    A = A.unsqueeze(-1)  # Shape: (batch_size, m, 1)
    B = B.unsqueeze(-2)  # Shape: (batch_size, 1, n)

    # Elementwise multiplication
    return A * B


def calculate_cov(loader, device, alpha=1):
    cumulative_cov = None
    cumulative_size = 0
    for data, _ in tqdm(loader, desc="Calculating Hessian", leave=True):
        data = data.to(device)
        if cumulative_cov is None:
            cumulative_cov = (data.T @ data).clone().detach().to('cpu')
        else:
            cumulative_cov += (data.T @ data).clone().detach().to('cpu')
        cumulative_size = cumulative_size + data.shape[0]
    return alpha * (cumulative_cov / cumulative_size)


def calculate_linear_ce_hess(model, loader, l2_reg=0.0, parallel=False, cov=False, alpha=1):
    param = list(model.parameters())[0]
    num_class, feat_dim = param.shape[0], param.shape[1]
    device = get_module_device(model)
    if parallel:
        total_H = None
        total_samples = 0

        with torch.no_grad():
            # Compute Hessian
            for batch_X, _ in tqdm(loader, desc="Calculating Hessian", leave=True):
                batch_X = batch_X.to(device)  # shape: [B, D]
                logits = model(batch_X)  # shape: [B, C]
                probs = torch.softmax(logits, dim=1)  # shape: [B, C]
                curr_size = batch_X.shape[0]

                softmax_grad = (
                        torch.diag_embed(probs)
                        - batched_outer(probs, probs)
                ).half().cpu().view(
                    curr_size,
                    2,
                    (num_class // 2),
                    2,
                    (num_class // 2)
                ).permute(0, 1, 3, 2, 4).contiguous().reshape(
                    curr_size,
                    4,
                    (num_class // 2),
                    (num_class // 2)
                )

                batch_H = torch.empty((curr_size, num_class * feat_dim, num_class * feat_dim), dtype=torch.float16)
                xx_t = batched_outer(batch_X, batch_X).half().to('cpu')

                def operation(idx):
                    try:
                        dev = 'cuda:{}'.format(idx) if torch.cuda.is_available() else 'cpu'
                        send = softmax_grad[:, idx, :].to(dev)
                        r, c = idx // 2, idx % 2
                        batch_H[:, (r * (num_class * feat_dim / 2)):((r + 1) * (num_class * feat_dim / 2)),
                        (c * (num_class * feat_dim / 2)):(
                                    (c + 1) * (num_class * feat_dim / 2))] = batched_kronecker_product(send, xx_t.to(
                            dev)).cpu()
                        return 1
                    except Exception as e:
                        print(e)
                        return 0

                futures = []
                for pid in range(4):
                    futures.append(torch.jit.fork(operation, pid))

                results = [torch.jit.wait(fut) for fut in futures]

                if total_H is None:
                    total_H = batch_H
                else:
                    total_H += batch_H
                total_samples += batch_X.shape[0]

        # Normalize Hessian by total number of samples and add L2 regularization
        total_H /= total_samples
        total_H = total_H.cpu()
        total_H += 2 * l2_reg * torch.eye(feat_dim * num_class)
        return total_H
    elif cov:
        device = get_module_device(model)
        block = calculate_cov(loader, device, alpha=alpha)
        return block + 2 * l2_reg * torch.eye(feat_dim)
    else:
        total_H = None
        total_samples = 0

        with torch.no_grad():
            # Compute Hessian
            for batch_X, _ in tqdm(loader, desc="Calculating Hessian", leave=True):
                batch_X = batch_X.to(device)  # shape: [B, D]
                logits = model(batch_X)  # shape: [B, C]
                probs = torch.softmax(logits, 1)  # shape: [B, C]
                B = batch_X.size(0)

                # -- Compute softmax Hessian term in class-space for the whole batch --
                # softmax_grad[b] = diag(p[b]) - p[b].outer(p[b]) for each sample in batch
                # Vectorized: [B, C, C]
                softmax_grad = (
                        torch.diag_embed(probs)
                        - torch.einsum('bi,bj->bij', probs, probs)
                )

                # -- Compute (X_i X_i^T) for each sample in batch, shape: [B, D, D] --
                # einsum('bi,bj->bij') or outer product per sample
                xx_t = torch.einsum('bi,bj->bij', batch_X, batch_X)

                # -- We want to sum of Kronecker( (D x D), (C x C) ) across all samples.
                #   Flatten (D x D) to D^2 and (C x C) to C^2, multiply, and reshape back.
                #   Then sum across B.
                # shape: [B, D*D]
                # xx_t_flat = xx_t.view(B, feat_dim * feat_dim)
                # shape: [B, C*C]
                # softmax_grad_flat = softmax_grad.view(B, num_class * num_class)

                # elementwise multiply + reshape => shape [B, (D*C)*(D*C)]
                # then sum over B
                # kron_batch = softmax_grad_flat.unsqueeze(2) * xx_t_flat.unsqueeze(1)
                # kron_batch shape: [B, D*D, C*C]
                softmax_grad = softmax_grad.to('cpu')
                xx_t = xx_t.to('cpu')
                kron_batch = softmax_grad.unsqueeze(2).unsqueeze(4) * xx_t.unsqueeze(1).unsqueeze(3)
                kron_batch = kron_batch.view(B, feat_dim * num_class, feat_dim * num_class)

                # Accumulate Hessian for this batch
                batch_H = kron_batch.sum(dim=0)  # sum over samples in the batch
                if total_H is None:
                    total_H = batch_H
                else:
                    total_H += batch_H
                total_samples += batch_X.shape[0]

        # Normalize Hessian by total number of samples and add L2 regularization
        total_H /= total_samples
        total_H += 2 * l2_reg * torch.eye(feat_dim * num_class)
        return total_H


def calculate_hessian(model, data_loader, criterion, save_path=None, linear=False, parallel=False, cov=False, alpha=1):
    """
    Compute the Hessian of the loss w.r.t. model parameters.
    """
    if linear:
        if hasattr(criterion, 'l2_lambda'):
            hessian_cpu = calculate_linear_ce_hess(model, data_loader, l2_reg=criterion.l2_lambda,
                                                   parallel=parallel, cov=cov, alpha=alpha)
        else:
            hessian_cpu = calculate_linear_ce_hess(model, data_loader, l2_reg=0,
                                                   parallel=parallel, cov=cov, alpha=alpha)
    else:
        device = get_module_device(model)
        params = list(model.parameters())
        num_params = sum(p.numel() for p in params)

        # Create an empty Hessian on CPU
        hessian_cpu = torch.zeros(num_params, num_params, dtype=torch.float32, device="cpu")
        num_samples = 0

        for batch in tqdm(data_loader, desc="Calculating Hessian", leave=True):
            x, y = batch
            x, y = x.to(device), y.to(device)
            model.zero_grad()

            # Forward pass
            logits = model(x)
            if isinstance(criterion, L2RegularizedCrossEntropyLoss):
                loss = criterion(logits, y, model)
            else:
                loss = criterion(logits, y)

            # First-order gradients
            grad_params = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            grad_params_flat = torch.cat([g.view(-1) for g in grad_params])

            # Second-order gradients
            batch_hessian_gpu = []
            for g in grad_params_flat:
                second_grad = torch.autograd.grad(
                    g, params, retain_graph=True, allow_unused=True
                )
                second_grad_flat = torch.cat([
                    sg.view(-1) if sg is not None else torch.zeros_like(p).view(-1)
                    for sg, p in zip(second_grad, params)
                ])
                batch_hessian_gpu.append(second_grad_flat)

            batch_hessian_gpu = torch.stack(batch_hessian_gpu)

            # Move the batch Hessian to CPU and accumulate
            batch_hessian_cpu = batch_hessian_gpu.detach().to("cpu")
            hessian_cpu = hessian_cpu + batch_hessian_cpu * x.shape[0]
            num_samples = num_samples + x.shape[0]

        # Average the Hessian
        hessian_cpu = hessian_cpu / num_samples

    if save_path is not None:
        torch.save(hessian_cpu, save_path)

    return hessian_cpu


# TODO: should be updated
def calculate_retain_hess(model, wloader, floader, criterion, save_path=None, surr=False, linear=False,
                          parallel=False, cov=False, alpha=1):
    if save_path is not None:
        if surr:
            whess_sp = os.path.join(save_path, "wshess.pt")
            fhess_sp = os.path.join(save_path, "fshess.pt")
        else:
            whess_sp = os.path.join(save_path, "whess.pt")
            fhess_sp = os.path.join(save_path, "fhess.pt")
    else:
        whess_sp = None
        fhess_sp = None
    whess = calculate_hessian(model, wloader, criterion, save_path=whess_sp, linear=linear, parallel=parallel, cov=cov,
                              alpha=alpha)
    fhess = calculate_hessian(model, floader, criterion, save_path=fhess_sp, linear=linear, parallel=parallel, cov=cov,
                              alpha=alpha)
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
        if isinstance(criterion, L2RegularizedCrossEntropyLoss):
            loss = criterion(output, label, model)
        else:
            loss = criterion(output, label)
        loss.backward()
        curr_grads = [param.grad.clone().detach().to('cpu') for param in model.parameters()]
        grads = _accumulate_grads(grads, curr_grads, total_size, total_size + data.shape[0])
        total_size = total_size + data.shape[0]
    return grads


def _linearize_grads(grads):
    flat_grad = [grad.view(-1) for grad in grads]
    prev_sizes = [grad.size() for grad in grads]
    return torch.cat(flat_grad), prev_sizes


def calculate_grad_norm(grad):
    grad, prev_sizes = _linearize_grads(grad)
    return min(torch.norm(grad).item(), 1), prev_sizes


def _adjust_update(update, prev_sizes):
    splits = []
    for size in prev_sizes:
        split = 1
        for s in size:
            split *= s
        splits.append(split)
    adjusted_update = torch.split(update, splits)
    adjusted_update = [adjusted_update[idx].view(prev_sizes[idx]) for idx in range(len(prev_sizes))]
    return adjusted_update


def calculate_update(hessian, grads, device, hess_ss, grad_ss, cov=False):
    flat_grad, prev_sizes = _linearize_grads(grads)
    flat_grad = flat_grad.to(device)
    eps = 1e-6
    hess = (hessian + eps * torch.eye(hessian.shape[0])).to(device)
    inv = torch.linalg.inv(hess)
    if cov:
        update = torch.concat([torch.mv(inv, sp).to('cpu') for sp in flat_grad.split(hess.shape[0])])
    else:
        update = torch.mv(inv, (grad_ss / hess_ss) * flat_grad).to('cpu')
    update = _adjust_update(update, prev_sizes)
    return update


def update_model(model, updates):
    params = [param for param in model.parameters()]
    device = get_module_device(model)
    with torch.no_grad():
        for param, update in zip(params, updates):
            param.add_(update.to(device))


######################################### NOISE
def approximate_upper_cross_entropy(loader, model, surr_model):
    device = get_module_device(model)
    required_probs = []
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            surr_output = surr_model(data)
            probs = softmax(output, dim=1)
            surr_probs = softmax(surr_output, dim=1)
            mask = one_hot(label.long(), num_classes=probs.size(1)).bool()
            required_probs.append(torch.div(surr_probs[mask], probs[mask]))
    return torch.sum(torch.log(torch.cat(required_probs))).item() / len(loader.dataset)


#######################################
# ENERGY FUNCTION: E(x) = -log p(y|x)
#######################################
def is_in_range(x, range_tensor):
    min_values, max_values = range_tensor[..., 0], range_tensor[..., 1]
    within_range = (x >= min_values) & (x <= max_values)
    return within_range.all().item()


def energy_function(model, x, T=5, range_tensor=None, range_penalty=0.5):
    # Forward pass
    logits = model(x)
    energy = -1 * T * torch.logsumexp(logits / T, dim=1)
    if range_tensor is not None:
        if is_in_range(x, range_tensor):
            energy += range_penalty
        else:
            energy -= range_penalty

    return -1 * energy[0]


#######################################
# LANGEVIN SAMPLING
#######################################
def langevin_sampling(model, input_size, max_iter=100, step_size=0.02, input_range=None,
                      range_tensor=None, range_penalty=0.5):
    if input_range is None:
        input_range = [-10, 10]
    device = get_module_device(model)
    x = torch.rand([1] + input_size, device=device, requires_grad=True)
    x.data = input_range[0] + (input_range[1] - (input_range[0])) * x.data

    j = 0
    while j < max_iter:
        # Zero-out gradient from previous step
        if x.grad is not None:
            x.grad.zero_()

        # Compute energy
        e = energy_function(model, x, range_tensor=range_tensor, range_penalty=range_penalty)

        # Backprop to get gradient
        e.backward()

        # Langevin update:
        with torch.no_grad():
            x = x - (step_size / 2) * x.grad + math.sqrt(step_size) * torch.randn_like(x)

        # Detach from graph to avoid accumulating history
        x = x.detach()
        x.requires_grad_()

        # (Optional) print intermediate status
        if (j + 1) % 1000 == 0:
            print(f"  Step {j + 1}, Energy: {-1 * e.item():.6f}")
        j += 1
    return x.detach().cpu()


def sample_from_exact_marginal(model, num_samples, input_size, batch_size,
                               max_iter=100, step_size=0.02, input_range=None,
                               range_tensor=None, range_penalty=0.5):
    """using energy functions and langevin dynamics"""
    # Generate samples via Langevin dynamics
    freeze_model(model)
    print("Generating samples with size {}...".format(input_size))
    samples = []
    for idx in range(num_samples):
        print('#########################################')
        samples.append(langevin_sampling(model, input_size, max_iter=max_iter,
                                         step_size=step_size, input_range=input_range,
                                         range_tensor=range_tensor, range_penalty=range_penalty))
        print('sample {} generated'.format(idx + 1))
        print('#########################################')
    samples = torch.cat(samples, dim=0)
    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size, shuffle=True)
    melt_model(model)
    logging.info(
        '{} number of samples with dimension {} are generated by initializing the input in {} and spending {} number of iterations for each sample with learning rate {}'.format(
            num_samples, input_size, input_range, max_iter, step_size))
    logging.info('generated sampled are placed in a loader with batch size: {}'.format(batch_size))
    return loader


def estimate_marginal_kl_distance(ploader, qloader, device):
    """using variational representation of the KL distance (Donsker Varadhan bound)"""
    ftheta, app_kl = train_dv_bound(ploader, qloader, device)
    return ftheta, app_kl


def brategnolle_huber(kl):
    return math.sqrt(1 - math.exp(-kl))


def calculate_upper_tv(known=False, surr_loader=None, model=None, surr_model=None, kl_distance=None, num_samples=500,
                       input_range=None):
    if known:
        app_up_cross = approximate_upper_cross_entropy(surr_loader, model, surr_model)
        app_kl = app_up_cross + kl_distance
        logging.info('app_up_cross: {}'.format(app_up_cross))
        logging.info('app_kl: {}'.format(app_kl))
        if app_kl < 0:
            app_kl = 0
        upper_tv = 2 * brategnolle_huber(app_kl)
        logging.info('upper_tv: {}'.format(upper_tv))
    elif surr_loader is not None:
        device = get_module_device(model)
        batch = next(iter(surr_loader))  # Get the first batch
        input_size = batch[0].shape[1:]
        gen_exact_loader = sample_from_exact_marginal(model, num_samples, input_size, surr_loader.batch_size,
                                                      input_range=input_range)
        _, app_kl_distance = estimate_marginal_kl_distance(surr_loader, gen_exact_loader, device)
        app_up_cross = approximate_upper_cross_entropy(surr_loader, model, surr_model)
        app_kl = app_up_cross + app_kl_distance
        upper_tv = 2 * brategnolle_huber(app_kl)
        print(upper_tv)
    else:
        upper_tv = None
    return upper_tv


def calculate_upper_app_unlearn_surr(grad, smooth, sc, known=False, surr_loader=None, model=None,
                                     surr_model=None, kl_distance=None):
    upper = smooth / (sc ** 2)
    upper_tv = calculate_upper_tv(known=known, surr_loader=surr_loader, model=model,
                                  surr_model=surr_model, kl_distance=kl_distance)
    grad_norm, prev_sizes = calculate_grad_norm(grad)
    if upper_tv is not None:
        upper = upper * upper_tv * grad_norm
        logging.info('upper: {}'.format(upper))
    else:
        upper = None
    return upper, prev_sizes


def calculate_upper_retrain_app_unlearn(num_retain, num_forget, sc, lip, hlip):
    numerator = 2 * hlip * (lip ** 2) * (num_forget ** 2)
    denominator = (sc ** 3) * (num_retain ** 2)
    return numerator / denominator


def set_noise(surr_size, forget_size, grad,
              eps, delta, smooth=1, sc=1, lip=1, hlip=1, surr=False,
              known=False, surr_loader=None, model=None,
              surr_model=None, kl_distance=None):
    upper_retrain_app_unlearn = calculate_upper_retrain_app_unlearn(surr_size,
                                                                    forget_size,
                                                                    sc, lip, hlip)
    logging.info('upper_retrain_app_unlearn: {}'.format(upper_retrain_app_unlearn))
    upper_app_unlearn_surr, prev_sizes = calculate_upper_app_unlearn_surr(grad, smooth, sc,
                                                                          known=known, surr_loader=surr_loader,
                                                                          model=model,
                                                                          surr_model=surr_model,
                                                                          kl_distance=kl_distance)
    if surr:
        upper = upper_retrain_app_unlearn + upper_app_unlearn_surr
    else:
        upper = upper_retrain_app_unlearn
    sigma = upper / eps
    sigma = sigma * math.sqrt(2 * math.log(1.25 / delta))
    logging.info(f"bound: {upper}, sigma: {sigma}")
    model_size = 0
    for size in prev_sizes:
        curr_size = 1
        for s in size:
            curr_size *= s
        model_size += curr_size

    update = torch.normal(0, sigma, size=(model_size,))
    update = _adjust_update(update, prev_sizes)
    return update


#########################################


def forget(model, whess_loader, fhess_loader, grad_loader, criterion, device, save_path=None,
           eps=None, delta=None, smooth=1, sc=1, lip=1, hlip=1, surr=False,
           known=False, surr_loader=None, surr_model=None, kl_distance=None, linear=False, parallel=False, cov=False,
           alpha=1):
    fmodel = deepcopy(model.to('cpu')).to(device)
    hess = calculate_retain_hess(fmodel, whess_loader, fhess_loader, criterion, save_path=save_path, surr=surr,
                                 linear=linear, parallel=parallel, cov=cov, alpha=alpha)
    grads = calculate_grad(fmodel, grad_loader, criterion)
    update = calculate_update(hess, grads, device, len(whess_loader.dataset) - len(fhess_loader.dataset),
                              len(grad_loader.dataset), cov=cov)
    update_model(fmodel, update)
    if eps is not None and delta is not None:
        model = model.to(device)
        noise = set_noise(len(whess_loader.dataset), len(fhess_loader.dataset), grads, eps, delta,
                          smooth=smooth, sc=sc, lip=lip, hlip=hlip, surr=surr,
                          known=known, surr_loader=surr_loader, model=model,
                          surr_model=surr_model, kl_distance=kl_distance)
        update_model(fmodel, noise)
    return fmodel
