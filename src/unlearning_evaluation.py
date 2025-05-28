"""Main file to launch unlearning evaluation using the competition unlearning_metric."""

import copy
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from src import unlearning_metric
import yaml
import math
from torchvision.models import resnet18, ResNet18_Weights

from src.forget import forget
from src.eval import evaluate
from src.train import train
from src.utils import get_module_device, set_seed

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_confs(net, loader):
    device = get_module_device(net)
    """Returns the confidences of the data in loader extracted from net."""
    confs = []
    for sample in loader:
        inputs, targets = sample
        inputs = inputs.to(device)
        logits = net(inputs)
        logits = logits.detach().cpu().numpy()
        _, conf = unlearning_metric.compute_logit_scaled_confidence(logits, targets)
        confs.append(conf)
    confs = np.concatenate(confs, axis=0)
    return confs


def real_return_model(model_config, dim, num_class):
    if model_config['type'] == 'mlp':
        bias = model_config['bias']
        if model_config['hidden_sizes'] is not None:
            model_arr = []
            curr_in = dim
            for size in model_config['hidden_sizes']:
                model_arr.append(nn.Linear(curr_in, size, bias=bias))
                if model_config['activation'] == 'relu':
                    model_arr.append(nn.ReLU())
                curr_in = size
            model_arr.append(nn.Linear(curr_in, num_class, bias=bias))
            model = nn.Sequential(*model_arr)
        else:
            model = nn.Linear(dim, num_class, bias=bias)
        return model
    elif model_config['type'] == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if model_config['mode'] == 'linear':
            model = nn.Sequential(nn.Flatten(),
                                  nn.Linear(model.fc.in_features, num_class))
        elif model_config['mode'] == 'conv1':
            model = nn.Sequential(
                model.layer4[1],  # Fourth residual block
                model.avgpool,  # Global average pooling
                nn.Flatten(),  # Flatten the tensor
                nn.Linear(model.fc.in_features, num_class)  # Fully connected layer
            )

            for idx, param in enumerate(model.parameters()):
                param.requires_grad = False
                if idx == 2:
                    break
        elif model_config['mode'] == 'conv2':
            model = nn.Sequential(
                model.layer4[1],  # Fourth residual block
                model.avgpool,  # Global average pooling
                nn.Flatten(),  # Flatten the tensor
                nn.Linear(model.fc.in_features, num_class)  # Fully connected layer
            )
        return model


def get_unlearned_and_retrained_confs_and_accs(
        original_model,
        smodel,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        surr_loader,
        retrained_confs_path,
        kl_distance,
        config_path,
        criterion,
        prev_size,
        sigma,
        unlearned_confs_path,
        num_models=512,
        device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """Returns the confidence and accuracies of unlearned and retrained models."""
    # Step 1) Get the confidences and accuracies under the unlearned models
    #######################################################################
    seeds = np.random.permutation(num_models)
    unlearned_confs_forget = []
    unlearned_retain_accs, unlearned_test_accs, unlearned_forget_accs = [], [], []
    recompute = True

    if os.path.exists(unlearned_confs_path):
        loaded_results = np.load(unlearned_confs_path)
        # retrained_confs is [num models, num examples].
        assert loaded_results['unlearned_confs'].shape[0] == num_models
        unlearned_confs_forget = loaded_results['unlearned_confs']
        unlearned_retain_accs = loaded_results['unlearned_retain_accs']
        unlearned_test_accs = loaded_results['unlearned_test_accs']
        unlearned_forget_accs = loaded_results['unlearned_forget_accs']
        recompute = False

    if recompute:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        unlearn_config = config['unlearn']
        eps_multiplier = unlearn_config['eps_multiplier']
        eps_power = unlearn_config['eps_power']
        delta = unlearn_config['delta']
        smooth = unlearn_config['smooth']
        sc = unlearn_config['sc']
        lip = unlearn_config['lip']
        hlip = unlearn_config['hlip']
        surr = unlearn_config['surr']
        known = unlearn_config['known']
        linear = unlearn_config['linear']
        parallel = unlearn_config['parallel']
        cov = unlearn_config['cov']
        if 'alpha' in unlearn_config.keys():
            alpha = unlearn_config['alpha']
        else:
            alpha = 1
        conjugate = unlearn_config['conjugate']
        eps = eps_multiplier * (math.e ** eps_power)
        for i in range(num_models):
            set_seed(int(seeds[i]))
            smodel = smodel.to(device)
            net = forget(original_model, surr_loader, forget_loader, forget_loader, criterion, device,
                         eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip, surr=surr,
                         known=known, surr_loader=surr_loader, surr_model=smodel, kl_distance=kl_distance,
                         linear=linear, parallel=parallel, cov=cov, alpha=alpha, conjugate=conjugate, prev_size=prev_size)
            net.eval()
            smodel = smodel.to('cpu')

            # For this particular model, compute the forget set confidences.
            confs_forget = _get_confs(net, forget_loader_no_shuffle)
            unlearned_confs_forget.append(confs_forget)
            # For this particular model, compute the retain and test accuracies.
            retain_acc = evaluate(retain_loader, net, criterion, device=device, log=True)
            test_acc = evaluate(test_loader, net, criterion, device=device, log=True)
            forget_acc = evaluate(forget_loader, net, criterion, device=device, log=True)
            unlearned_retain_accs.append(retain_acc)
            unlearned_test_accs.append(test_acc)
            unlearned_forget_accs.append(forget_acc)

        unlearned_confs_forget = np.stack(unlearned_confs_forget)
        np.savez(
            unlearned_confs_path,
            unlearned_confs=unlearned_confs_forget,
            unlearned_retain_accs=unlearned_retain_accs,
            unlearned_test_accs=unlearned_test_accs,
            unlearned_forget_accs=unlearned_forget_accs,
        )

    # Step 2) Get the confidences and accuracies under the retrained models
    #######################################################################
    recompute = True
    retrained_confs_forget = []
    retrain_retain_accs, retrain_test_accs, retrain_forget_accs = [], [], []

    if os.path.exists(retrained_confs_path):
        loaded_results = np.load(retrained_confs_path)
        # retrained_confs is [num models, num examples].
        assert loaded_results['retrained_confs'].shape[0] == num_models
        retrained_confs_forget = loaded_results['retrained_confs']
        retrain_retain_accs = loaded_results['retrain_retain_accs']
        retrain_test_accs = loaded_results['retrain_test_accs']
        retrain_forget_accs = loaded_results['retrain_forget_accs']
        recompute = False

    if recompute:
        model_config = config['train']['model']
        dim = config['data']['dim']
        num_class = config['data']['num_class']
        for i in range(num_models):
            set_seed(int(seeds[i]))
            net = real_return_model(model_config, dim, num_class)
            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=config['train']['lr'])
            train(retain_loader, test_loader, net, criterion, optimizer, num_epoch=10, device=device)
            for param in net.parameters():
                if param.requires_grad:
                    param.data += (torch.randn_like(param) * sigma)

            # For this particular model, compute the forget set confidences.
            confs_forget = _get_confs(net, forget_loader_no_shuffle)
            retrained_confs_forget.append(confs_forget)

            retain_acc = evaluate(retain_loader, net, criterion, device=device, log=True)
            test_acc = evaluate(test_loader, net, criterion, device=device, log=True)
            forget_acc = evaluate(forget_loader, net, criterion, device=device, log=True)
            retrain_retain_accs.append(retain_acc)
            retrain_test_accs.append(test_acc)
            retrain_forget_accs.append(forget_acc)

        retrained_confs_forget = np.stack(retrained_confs_forget)

        np.savez(
            retrained_confs_path,
            retrained_confs=retrained_confs_forget,
            retrain_retain_accs=retrain_retain_accs,
            retrain_test_accs=retrain_test_accs,
            retrain_forget_accs=retrain_forget_accs,
        )

    return (
        unlearned_confs_forget,
        retrained_confs_forget,
        unlearned_retain_accs,
        unlearned_test_accs,
        unlearned_forget_accs,
        retrain_retain_accs,
        retrain_test_accs,
        retrain_forget_accs,
    )


def final_score(
        original_model,
        smodel,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        surr_loader,
        retrained_confs_path,
        kl_distance,
        config_path,
        criterion,
        prev_size,
        sigma,
        unlearned_confs_path,
        num_models=512,
        device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Running on device: %s', device)
    print('torch version: %s', torch.__version__)
    (
        unlearned_confs_forget,
        retrained_confs_forget,
        unlearned_retain_accs,
        unlearned_test_accs,
        unlearned_forget_accs,
        retrain_retain_accs,
        retrain_test_accs,
        retrain_forget_accs,
    ) = get_unlearned_and_retrained_confs_and_accs(
        original_model,
        smodel,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        surr_loader,
        retrained_confs_path,
        kl_distance,
        config_path,
        criterion,
        prev_size,
        sigma,
        unlearned_confs_path,
        num_models=num_models,
        device=device
    )

    u_r_mean = np.mean(unlearned_retain_accs)
    u_t_mean = np.mean(unlearned_test_accs)
    u_f_mean = np.mean(unlearned_forget_accs)
    r_r_mean = np.mean(retrain_retain_accs)
    r_t_mean = np.mean(retrain_test_accs)
    r_f_mean = np.mean(retrain_forget_accs)

    forget_score = unlearning_metric.compute_forget_score_from_confs(
        unlearned_confs_forget, retrained_confs_forget
    )

    final_score = forget_score * (u_r_mean / r_r_mean) * (u_t_mean / r_t_mean)
    return final_score, forget_score