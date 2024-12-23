# WARNING: BECAUSE OF CIRCULAR IMPORT MOVED TO FORGET DIRECTLY


from torch.nn.functional import softmax, one_hot
import torch
import argparse
import math

from src.utils import get_module_device
from remedi.train_complete import train_complete_model
from src.forget import calculate_grad_norm, update_model, _adjust_update



def approximate_upper_cross_entropy(loader, model):
    device = get_module_device(model)
    required_probs = []
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        probs = softmax(output, dim=1)
        mask = one_hot(label.long(), num_classes=probs.size(0)).bool()
        required_probs.append(probs[mask])
    return torch.sum(torch.log(torch.cat(required_probs))).item() / len(loader.dataset)


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
                                     tighten_bound, train_dataset, val_dataset):
    upper = smooth / (sc ** 2)
    upper_tv = calculate_upper_tv(surr_loader, model, tighten_bound, train_dataset, val_dataset)
    grad_norm, prev_sizes = calculate_grad_norm(grad)
    upper = upper * upper_tv * grad_norm
    return upper, prev_sizes


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
                                                                          tighten_bound, train_dataset, val_dataset)
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
