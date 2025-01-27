#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
# from argparse import Namespace
import argparse
import yaml
import torch.nn as nn
import os
import numpy as np
import math
from torchvision.models import resnet18, ResNet18_Weights

from src.utils import set_seed
from src.synthetic import GaussianDataset
from src.data import get_retain_forget_datasets, get_dataloaders, get_transforms, get_train_test_datasets, \
    get_exact_surr_datasets
from src.loss import L2RegularizedCrossEntropyLoss
from src.forget import forget
from src.metrics import membership_inference_attack, relearn_time
from src.eval import evaluate
from torch.utils.data import DataLoader
from src.unlearning_evaluation import final_score


# In[2]:


def return_model(model_config, dim, num_class):
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


def replace_none_with_none(d):
    for k, v in d.items():
        if isinstance(v, dict):
            replace_none_with_none(v)
        elif v == 'none':
            d[k] = None


def log_eval(model, train_loader, val_loader, retain_loader, forget_loader, surr_loader, criterion, device):
    train_acc = evaluate(train_loader, model, criterion, device=device, log=True)
    test_acc = evaluate(val_loader, model, criterion, device=device, log=True)
    retain_acc = evaluate(retain_loader, model, criterion, device=device, log=True)
    forget_acc = evaluate(forget_loader, model, criterion, device=device, log=True)
    surr_acc = evaluate(surr_loader, model, criterion, device=device, log=True)
    print(
        'train: {}, test: {}, retain: {}, forget: {}, surrogate:{}'.format(train_acc, test_acc, retain_acc, forget_acc,
                                                                           surr_acc))
    return forget_acc


# In[3]:

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wrapper of all real forget dataset experiments')
    parser.add_argument('--base', type=str, required=True, help='path to base folder')
    parser.add_argument('--device', type=int, required=True, help='device number')
    args = parser.parse_args()
    config_path = os.path.join(args.base, 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    replace_none_with_none(config)
    base_save_dir = config['setup']['base_save_dir']
    about = config['setup']['about']

    # set seed
    seed = config['setup']['seed']
    set_seed(seed)

    # set device
    device_id = args.device
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'

    # set data
    data_config = config['data']
    exact_dataset_type = data_config['exact_dataset']
    surrogate_dataset_type = data_config['surrogate_dataset']
    num_samples = data_config['num_samples']
    dim = data_config['dim']
    exact_mean = np.zeros(dim)
    exact_cov = np.eye(dim)
    num_class = data_config['num_class']
    off_cov = data_config['off_cov']
    off_cov = data_config['off_cov']
    if exact_dataset_type == 'gaussian':
        dataset = GaussianDataset(num_samples, num_class, exact_mean, exact_cov)

    if surrogate_dataset_type == 'gaussian' and exact_dataset_type == 'gaussian':
        surr_cov = exact_cov + off_cov * (np.ones_like(exact_cov) - np.eye(dim))
        surr_dataset = dataset.create_surr(exact_mean, surr_cov)
        kl_distance = surr_dataset.calculate_kl_between(dataset)

    train_config = config['train']

    # set train test data
    test_ratio = train_config['test_ratio']
    train_dataset, test_dataset = get_retain_forget_datasets(dataset, test_ratio)
    forget_ratio = config['unlearn']['forget_ratio']
    retain_dataset, forget_dataset = get_retain_forget_datasets(train_dataset, forget_ratio)
    train_loader, test_loader = get_dataloaders([train_dataset, test_dataset], train_config['batch_size'])
    retain_loader, forget_loader = get_dataloaders([retain_dataset, forget_dataset], train_config['batch_size'])
    surr_loader = get_dataloaders(surr_dataset, train_config['batch_size'])

    lambda_param = train_config['lambda']
    criterion = L2RegularizedCrossEntropyLoss(lambda_param)

    # set model
    model_config = train_config['model']
    model = return_model(model_config, dim, num_class)
    model.load_state_dict(torch.load(os.path.join(args.base, 'initial_model.pth'), weights_only=False))
    model = model.to(device)
    target_acc = log_eval(model, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion,
                          device)
    model = model.to('cpu')

    # In[4]:

    smodel = return_model(model_config, dim, num_class)
    smodel.load_state_dict(torch.load(os.path.join(args.base, 'surrogate_model.pth'), weights_only=False))
    smodel = smodel.to(device)
    log_eval(smodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    smodel = smodel.to('cpu')

    # In[5]:

    floader_nosh = DataLoader(forget_dataset, batch_size=train_config['batch_size'], shuffle=False)
    retrained_conf_path_less = os.path.join(args.base, 'retrained_conf_less.npz')
    retrained_conf_path_more = os.path.join(args.base, 'retrained_conf_more.npz')
    retrained_conf_path_retrain = os.path.join(args.base, 'retrained_conf_retrain.npz')
    unlearned_conf_path_less = os.path.join(args.base, 'unlearned_conf_less.npz')
    unlearned_conf_path_more = os.path.join(args.base, 'unlearned_conf_more.npz')
    unlearned_conf_path_retrain = os.path.join(args.base, 'unlearned_conf_retrain.npz')

    sigma_arr = []
    with open(os.path.join(args.base, 'experiment.log'), 'r') as f:
        for line in f:
            if 'INFO:root:bound:' in line and 'sigma' in line:
                sigma_arr.append(float(line.split(',')[-1].split()[-1]))

    sigma_less, sigma_more = sigma_arr[0], sigma_arr[1]
    # In[6]:

    fscore, forget_score = final_score(model, smodel, test_loader, retain_loader, forget_loader, floader_nosh,
                                       surr_loader,
                                       retrained_conf_path_more, kl_distance, config_path, criterion,
                                       len(train_dataset),
                                       sigma_more, unlearned_conf_path_more, device=device, num_models=512)

    # In[7]:

    fscore2, forget_score2 = final_score(model, smodel, test_loader, retain_loader, forget_loader, floader_nosh,
                                         surr_loader, retrained_conf_path_less, 0, config_path, criterion,
                                         len(train_dataset), sigma_less, unlearned_conf_path_less, device=device,
                                         num_models=100)

    fscore3, forget_score3 = final_score(model, smodel, test_loader, retain_loader, forget_loader, floader_nosh,
                                         train_loader, retrained_conf_path_retrain, 0, config_path, criterion,
                                         len(train_dataset), sigma_less, unlearned_conf_path_retrain, device=device,
                                         num_models=100)

    # In[8]:

    print('fscore with kl distance in play:', forget_score)
    print('fscore with kl distance not in play:', forget_score2)
    print('fscore with kl distance not required:', forget_score3)

    with open(os.path.join(args.base, 'forget_scores.yaml'), 'w') as file:
        yaml.safe_dump({
            'score-+': forget_score,
            'score--': forget_score2,
            'score++': forget_score3,
        }, file)
