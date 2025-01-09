import argparse
import yaml
import os
from datetime import datetime
import logging
import math

import torch
import torch.nn as nn
import numpy as np

from src.utils import set_seed
from src.synthetic import GaussianDataset
from src.data import get_retain_forget_datasets, get_dataloaders
from src.loss import L2RegularizedCrossEntropyLoss
from src.train import train
from src.eval import evaluate
from src.forget import forget


def log_eval(model, train_loader, val_loader, retain_loader, forget_loader, surr_loader, criterion, device):
    train_acc = evaluate(train_loader, model, criterion, device=device, log=True)
    test_acc = evaluate(val_loader, model, criterion, device=device, log=True)
    retain_acc = evaluate(retain_loader, model, criterion, device=device, log=True)
    forget_acc = evaluate(forget_loader, model, criterion, device=device, log=True)
    surr_acc = evaluate(surr_loader, model, criterion, device=device, log=True)
    logging.info(
        'train: {}, test: {}, retain: {}, forget: {}, surrogate:{}'.format(train_acc, test_acc, retain_acc, forget_acc,
                                                                           surr_acc))


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


def main():
    parser = argparse.ArgumentParser(description='wrapper of all synthetic experiments')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    # read config set experiment
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    replace_none_with_none(config)
    base_save_dir = config['setup']['base_save_dir']
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    now = datetime.now()
    experiment_dir = os.path.join(base_save_dir, now.strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(experiment_dir)
    logging.basicConfig(filename=os.path.join(experiment_dir, 'experiment.log'), level=logging.INFO)
    logging.info('experiment started at %s', now.strftime('%Y-%m-%d %H:%M:%S'))

    # copy config
    config_copy_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_copy_path, 'w') as file:
        yaml.safe_dump(config, file)
    logging.info('configuration file copied to %s', config_copy_path)

    # set seed
    seed = config['setup']['seed']
    set_seed(seed)
    logging.info('seed: %s', seed)

    # set device
    device_id = config['setup']['device']
    device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
    logging.info('device: %s', device)

    # set data
    logging.info('#####################')
    logging.info('setting data')
    data_config = config['data']
    exact_dataset_type = data_config['exact_dataset']
    surrogate_dataset_type = data_config['surrogate_dataset']
    num_samples = data_config['num_samples']
    dim = data_config['dim']
    exact_mean = np.zeros(dim)
    exact_cov = np.eye(dim)
    num_class = data_config['num_class']
    off_cov = data_config['off_cov']
    if exact_dataset_type == 'gaussian':
        dataset = GaussianDataset(num_samples, num_class, exact_mean, exact_cov)

    if surrogate_dataset_type == 'gaussian' and exact_dataset_type == 'gaussian':
        surr_cov = exact_cov + off_cov * (np.ones_like(exact_cov) - np.eye(dim))
        surr_dataset = dataset.create_surr(exact_mean, surr_cov)
        kl_distance = surr_dataset.calculate_kl_between(dataset)
    logging.info('exact and surrogate dataset created')
    logging.info('exact dataset size: {}, dim: {}'.format(num_samples, dim))
    logging.info('surrogate dataset size: {}, dim: {}, off_cov: {}'.format(num_samples, dim, off_cov))
    logging.info('kl distance between surrogate and exact marginal distributions: {}'.format(kl_distance))
    logging.info('#####################')

    logging.info('#####################')
    logging.info('training setup')
    # train
    train_config = config['train']

    # set train test data
    test_ratio = train_config['test_ratio']
    train_dataset, test_dataset = get_retain_forget_datasets(dataset, test_ratio)
    forget_ratio = config['unlearn']['forget_ratio']
    retain_dataset, forget_dataset = get_retain_forget_datasets(train_dataset, forget_ratio)
    train_loader, test_loader = get_dataloaders([train_dataset, test_dataset], train_config['batch_size'])
    retain_loader, forget_loader = get_dataloaders([retain_dataset, forget_dataset], train_config['batch_size'])
    surr_loader = get_dataloaders(surr_dataset, train_config['batch_size'])

    logging.info('all dataloaders created')
    logging.info('test ratio: {}, train dataset size: {}, test dataset size: {}'.format(test_ratio, len(train_dataset),
                                                                                        len(test_dataset)))
    logging.info(
        'forget ratio: {}, retain dataset size: {}, forget dataset size: {}'.format(forget_ratio, len(retain_dataset),
                                                                                    len(forget_dataset)))

    lambda_param = train_config['lambda']
    criterion = L2RegularizedCrossEntropyLoss(lambda_param)
    logging.info('criterion: {}, lambda: {}'.format(criterion, lambda_param))

    # set model
    model_config = train_config['model']
    model = return_model(model_config, dim, num_class)
    model = model.to(device)
    logging.info('model: {}'.format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    logging.info('lr: {}'.format(train_config['lr']))
    logging.info('#####################')

    # train
    num_epochs = train_config['num_epochs']
    logging.info('#####################')
    logging.info('INITIAL TRAINING')
    train(train_loader, test_loader, model, criterion, optimizer, num_epoch=num_epochs, device=device)
    log_eval(model, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    model = model.to('cpu')
    # save model state dict
    model_save_path = os.path.join(experiment_dir, 'initial_model.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info('initial model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # retrain
    logging.info('#####################')
    logging.info('RETRAIN FROM SCRATCH')
    rmodel = return_model(model_config, dim, num_class)
    rmodel = rmodel.to(device)
    optimizer = torch.optim.Adam(rmodel.parameters(), lr=train_config['lr'])
    train(retain_loader, test_loader, rmodel, criterion, optimizer, num_epoch=num_epochs, device=device)
    log_eval(rmodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    rmodel = rmodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'retrained_model.pth')
    torch.save(rmodel.state_dict(), model_save_path)
    logging.info('retrained model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # surrogate training
    logging.info('#####################')
    logging.info('SURROGATE MODEL TRAINING')
    smodel = return_model(model_config, dim, num_class)
    smodel = smodel.to(device)
    optimizer = torch.optim.Adam(smodel.parameters(), lr=train_config['lr'])
    train(surr_loader, test_loader, smodel, criterion, optimizer, num_epoch=num_epochs, device=device)
    log_eval(smodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    smodel = smodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'surrogate_model.pth')
    torch.save(smodel.state_dict(), model_save_path)
    logging.info('surrogate model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # unlearn config
    unlearn_config = config['unlearn']
    forget_ratio = unlearn_config['forget_ratio']
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

    # unlearn with exact
    logging.info('#####################')
    logging.info('UNLEARN WITH EXACT')
    logging.info('noise --> eps_multiplier: {}, eps_power: {}, delta: {}, smooth: {}, sc: {}, lip: {}, hlip: {}'.format(
        eps_multiplier, eps_power, delta, smooth, sc, lip, hlip))
    eps = eps_multiplier * (math.e ** eps_power)
    umodel = forget(model, train_loader, forget_loader, forget_loader, criterion, device, save_path=experiment_dir,
                    eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip,
                    linear=linear, parallel=parallel, cov=cov)
    log_eval(umodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    umodel = umodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'uexact_model.pth')
    torch.save(umodel.state_dict(), model_save_path)
    logging.info('unlearn with exact model state dict saved to %s', model_save_path)
    logging.info('#####################')

    if surr:
        # unlearn with surrogate
        logging.info('#####################')
        logging.info('UNLEARN WITH SURROGATE')
        logging.info(
            'noise --> eps_multiplier: {}, eps_power: {}, delta: {}, smooth: {}, sc: {}, lip: {}, hlip: {}, kl_distance: {}'.format(
                eps_multiplier, eps_power, delta, smooth, sc, lip, hlip, kl_distance))
        smodel = smodel.to(device)
        usmodel = forget(model, surr_loader, forget_loader, forget_loader, criterion, device, save_path=experiment_dir,
                         eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip, surr=surr,
                         known=known, surr_loader=surr_loader, surr_model=smodel, kl_distance=kl_distance,
                         linear=linear, parallel=parallel, cov=cov)
        log_eval(usmodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
        usmodel = usmodel.to('cpu')
        smodel = smodel.to('cpu')
        model_save_path = os.path.join(experiment_dir, 'usurr_model.pth')
        torch.save(usmodel.state_dict(), model_save_path)
        logging.info('unlearn with surrogate model state dict saved to %s', model_save_path)
        logging.info('#####################')


if __name__ == '__main__':
    main()
