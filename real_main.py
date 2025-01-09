import argparse
import yaml
import os
from datetime import datetime
import logging
import math

import torch
import torch.nn as nn

from src.utils import set_seed
from src.data import (get_retain_forget_datasets, get_dataloaders, get_train_test_datasets,
                      get_transforms, get_exact_surr_datasets)
from src.loss import L2RegularizedCrossEntropyLoss
from src.train import train
from src.eval import evaluate
from src.forget import forget, sample_from_exact_marginal, estimate_marginal_kl_distance
from src.metrics import membership_inference_attack, relearn_time


def log_eval(model, train_loader, val_loader, retain_loader, forget_loader, surr_loader, criterion, device):
    train_acc = evaluate(train_loader, model, criterion, device=device, log=True)
    test_acc = evaluate(val_loader, model, criterion, device=device, log=True)
    retain_acc = evaluate(retain_loader, model, criterion, device=device, log=True)
    forget_acc = evaluate(forget_loader, model, criterion, device=device, log=True)
    surr_acc = evaluate(surr_loader, model, criterion, device=device, log=True)
    logging.info(
        'train: {}, test: {}, retain: {}, forget: {}, surrogate:{}'.format(train_acc, test_acc, retain_acc, forget_acc,
                                                                           surr_acc))
    return train_acc


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
    parser = argparse.ArgumentParser(description='wrapper of all real dataset experiments')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    # read config set experiment
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    replace_none_with_none(config)
    base_save_dir = config['setup']['base_save_dir']
    about = config['setup']['about']
    curr_dict = config
    for key in about.split('-'):
        curr_dict = curr_dict[key]
    about_value = curr_dict
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    now = datetime.now()
    experiment_dir = os.path.join(base_save_dir, '{}-{}-{}'.format(about,
                                                                   str(about_value),
                                                                   now.strftime('%Y-%m-%d-%H-%M-%S')))
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
    dim = data_config['dim']
    num_class = data_config['num_class']
    transforms = get_transforms(exact_dataset_type)
    train_dataset, test_dataset = get_train_test_datasets(exact_dataset_type, transform=transforms,
                                                         train_path=data_config['train_path'],
                                                         test_path=data_config['test_path'],
                                                         save_path=data_config['save_path'],
                                                         device=device)
    if surrogate_dataset_type is not None:
        stransforms = get_transforms(surrogate_dataset_type)
        surr_dataset, _ = get_train_test_datasets(surrogate_dataset_type, transform=transforms,
                                                  train_path=data_config['strain_path'],
                                                  test_path=data_config['stest_path'],
                                                  save_path=data_config['ssave_path'],
                                                  device=device)
    else:
        exact_size = int(len(train_dataset) / 2)
        surr_size = len(train_dataset) - exact_size
        dirichlet = data_config['dirichlet']
        train_dataset, surr_dataset = get_exact_surr_datasets(train_dataset,
                                                              target_size=exact_size,
                                                              starget_size=surr_size,
                                                              dirichlet=dirichlet, num_class=num_class)

    logging.info('exact and surrogate dataset created')
    logging.info('exact dataset size: {}, dim: {}'.format(len(train_dataset), dim))
    logging.info('surrogate dataset size: {}, dim: {}'.format(len(surr_dataset), dim))
    logging.info('#####################')

    logging.info('#####################')
    logging.info('training setup')
    # train
    train_config = config['train']

    # set train test data
    forget_ratio = config['unlearn']['forget_ratio']
    retain_dataset, forget_dataset = get_retain_forget_datasets(train_dataset, forget_ratio)
    train_loader, test_loader = get_dataloaders([train_dataset, test_dataset], train_config['batch_size'])
    retain_loader, forget_loader = get_dataloaders([retain_dataset, forget_dataset], train_config['batch_size'])
    surr_loader = get_dataloaders(surr_dataset, train_config['batch_size'])

    logging.info('all dataloaders created')
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

    langevin_config = config['langevin']

    # train
    num_epochs = train_config['num_epochs']
    logging.info('#####################')
    logging.info('INITIAL TRAINING')
    train(train_loader, test_loader, model, criterion, optimizer, num_epoch=num_epochs, device=device)
    target_acc = log_eval(model, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    egensample_loader = sample_from_exact_marginal(model, langevin_config['num_samples'],
                                                   langevin_config['input_size'],
                                                   train_config['batch_size'],
                                                   input_range=langevin_config['input_range'],
                                                   max_iter=langevin_config['max_iter'],
                                                   step_size=langevin_config['step_size'])
    model = model.to('cpu')
    # save model state dict
    model_save_path = os.path.join(experiment_dir, 'initial_model.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info('initial model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # surrogate training
    logging.info('#####################')
    logging.info('SURROGATE MODEL TRAINING')
    smodel = return_model(model_config, dim, num_class)
    smodel = smodel.to(device)
    optimizer = torch.optim.Adam(smodel.parameters(), lr=train_config['lr'])
    train(surr_loader, test_loader, smodel, criterion, optimizer, num_epoch=num_epochs, device=device)
    log_eval(smodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    sgensample_loader = sample_from_exact_marginal(smodel, langevin_config['num_samples'],
                                                   langevin_config['input_size'],
                                                   train_config['batch_size'],
                                                   input_range=langevin_config['input_range'],
                                                   max_iter=langevin_config['max_iter'],
                                                   step_size=langevin_config['step_size'])
    smodel = smodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'surrogate_model.pth')
    torch.save(smodel.state_dict(), model_save_path)
    logging.info('surrogate model state dict saved to %s', model_save_path)
    logging.info('#####################')

    # kl distance estimation
    _, kl_distance = estimate_marginal_kl_distance(sgensample_loader, egensample_loader, device)
    _.to('cpu')
    del _

    logging.info('kl distance estimated using generated samples is {}'.format(kl_distance))

    # retrain
    logging.info('#####################')
    logging.info('RETRAIN FROM SCRATCH')
    rmodel = return_model(model_config, dim, num_class)
    rmodel = rmodel.to(device)
    optimizer = torch.optim.Adam(rmodel.parameters(), lr=train_config['lr'])
    train(retain_loader, test_loader, rmodel, criterion, optimizer, num_epoch=num_epochs, device=device)
    log_eval(rmodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    mia_score = membership_inference_attack(rmodel, test_loader, forget_loader)
    logging.info('MIA {}'.format(mia_score))
    required_iters = relearn_time(rmodel, criterion, train_loader, forget_loader, lr=train_config['lr'],
                                  target_acc=target_acc)
    logging.info('relearn time T {}'.format(required_iters))
    rmodel = rmodel.to('cpu')
    model_save_path = os.path.join(experiment_dir, 'retrained_model.pth')
    torch.save(rmodel.state_dict(), model_save_path)
    logging.info('retrained model state dict saved to %s', model_save_path)
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
                    eps=eps, delta=delta, smooth=smooth, sc=sc, lip=lip, hlip=hlip, linear=linear,
                    parallel=parallel, cov=cov)
    log_eval(umodel, train_loader, test_loader, retain_loader, forget_loader, surr_loader, criterion, device)
    mia_score = membership_inference_attack(umodel, test_loader, forget_loader)
    logging.info('MIA {}'.format(mia_score))
    required_iters = relearn_time(umodel, criterion, train_loader, forget_loader, lr=train_config['lr'],
                                  target_acc=target_acc)
    logging.info('relearn time T {}'.format(required_iters))
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
        smodel = smodel.to('cpu')
        mia_score = membership_inference_attack(usmodel, test_loader, forget_loader)
        logging.info('MIA {}'.format(mia_score))
        required_iters = relearn_time(usmodel, criterion, train_loader, forget_loader, lr=train_config['lr'],
                                      target_acc=target_acc)
        logging.info('relearn time T {}'.format(required_iters))
        usmodel = usmodel.to('cpu')
        model_save_path = os.path.join(experiment_dir, 'usurr_model.pth')
        torch.save(usmodel.state_dict(), model_save_path)
        logging.info('unlearn with surrogate model state dict saved to %s', model_save_path)
        logging.info('#####################')


if __name__ == '__main__':
    main()
