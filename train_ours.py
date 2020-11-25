"""
    (Adversarially) Train multi-exit architectures
"""
import os, re, json
import argparse

import platform
from datetime import datetime

# torch libs
import torch

# custom libs
import datasets, models, utils
from scenarios import scenario_1_split, scenario_2_split


"""
    Training functions (for CNNs and SDNs)
"""
def train(networks, storedir, sdn=False, device='cpu'):
    print('[Train] start training...')

    # loop over the networks
    for each_network in networks:

        # : load the initial network
        network, parameters = models.load_model(storedir, each_network, 0)

        # : use cuda (or not)
        network.to(device)

        # : load the dataset
        dataset = utils.load_dataset(parameters['task'], doNormalization=False)

        # : set the optimizer
        learning_rate = parameters['learning_rate']
        momentum      = parameters['momentum']
        weight_decay  = parameters['weight_decay']
        milestones    = parameters['milestones']
        gammas        = parameters['gammas']
        num_epochs    = parameters['epochs']
        parameters['optimizer'] = 'SGD'

        # : set the optimizers for IC-only training
        if sdn:
            learning_rate = parameters['ic_only']['learning_rate']
            milestones    = parameters['ic_only']['milestones']
            gammas        = parameters['ic_only']['gammas']
            num_epochs    = parameters['ic_only']['epochs']
            parameters['optimizer'] = 'Adam'

            # :: IC-only flag to the model
            network.ic_only = True

        # : set the optimizer parameters
        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        # : load the optimizers
        if sdn:
            # :: SDN training (IC-only or training from scratch)
            optimizer, scheduler = utils.load_sdn_ic_only_optimizer(network, optimization_params, lr_schedule_params)
            network_name = each_network
        else:
            optimizer, scheduler = utils.load_optimizer(network, optimization_params, lr_schedule_params)
            network_name = each_network

        print('[Train] start...')
        metrics = network.train_func( \
            network, dataset, num_epochs, optimizer, scheduler, None, device=device)

        # : store the validation metrics
        parameters['train_top1_acc'] = metrics['train_top1_acc']
        parameters['test_top1_acc']  = metrics['test_top1_acc']
        parameters['train_top5_acc'] = metrics['train_top5_acc']
        parameters['test_top5_acc']  = metrics['test_top5_acc']
        parameters['epoch_times']    = metrics['epoch_times']
        parameters['lrs']            = metrics['lrs']
        total_training_time          = sum(parameters['epoch_times'])
        parameters['total_time']     = total_training_time
        print('[Train] take {} seconds...'.format(total_training_time))

        # : save the model
        models.save_model(network, network_name, parameters, storedir, epoch=-1)
    # done.

def adv_train( \
    networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=False, device='cpu'):
    print('[Adv-Train] start training...')

    # loop over the networks
    for each_network in networks:

        # : load the initial network
        network, parameters = models.load_model(storedir, each_network, 0)

        # : use cuda (or not)
        network.to(device)

        # : load the dataset
        dataset = utils.load_dataset(parameters['task'], doNormalization=False)

        # : set the optimizer
        learning_rate = parameters['learning_rate']
        momentum      = parameters['momentum']
        weight_decay  = parameters['weight_decay']
        milestones    = parameters['milestones']
        gammas        = parameters['gammas']
        num_epochs    = parameters['epochs']
        parameters['optimizer'] = 'SGD'

        # : set the optimizers for IC-only training
        if sdn:
            learning_rate = parameters['ic_only']['learning_rate']
            num_epochs    = parameters['ic_only']['epochs']
            milestones    = parameters['ic_only']['milestones']
            gammas        = parameters['ic_only']['gammas']
            parameters['optimizer'] = 'Adam'

            # :: IC-only flag to the model
            network.ic_only = True

        # : set the optimizer parameters
        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        # : load the optimizers
        if sdn:
            # :: SDN training (IC-only or training from scratch)
            optimizer, scheduler = utils.load_sdn_ic_only_optimizer(network, optimization_params, lr_schedule_params)
            network_name = each_network

        else:
            optimizer, scheduler = utils.load_optimizer(network, optimization_params, lr_schedule_params)
            network_name = each_network

        print ('[Adv-Train] start...')
        metrics = network.advtrain_func( \
            network, dataset, num_epochs, optimizer, scheduler, None, \
            attack, max_iter, eps_step, epsilon, device=device)

        # : store the validation metrics
        parameters['train_top1_acc'] = metrics['train_top1_acc']
        parameters['test_top1_acc']  = metrics['test_top1_acc']
        parameters['train_top5_acc'] = metrics['train_top5_acc']
        parameters['test_top5_acc']  = metrics['test_top5_acc']
        parameters['epoch_times']    = metrics['epoch_times']
        parameters['lrs']            = metrics['lrs']
        total_training_time          = sum(parameters['epoch_times'])
        parameters['total_time']     = total_training_time
        print('[Adv-Train] take {} seconds...'.format(total_training_time))

        # : save the model
        models.save_model(network, network_name, parameters, storedir, epoch=-1)
    # done.

def train_sdns(networks, storedir, sdn=True, device='cpu'):
    # training strategies
    load_epoch = -1     # IC-only

    # loop over the networks, and set the training configurations
    for each_network in networks:
        cnn_to_tune = re.sub("sdn$", "cnn", each_network)

        # Added by ionmodo
        # because of the above line, the dictionary containing hyperparameters of the CNN will contain
        # the parameter called 'doNormalization' with value False set in create_vgg16bn in network_architectures
        sdn_params = models.load_params(storedir, each_network)
        sdn_params = models.load_cnn_parameters(sdn_params['task'], sdn_params['network_type'])
        sdn_model, _ = utils.cnn_to_sdn(storedir, cnn_to_tune, sdn_params, load_epoch)
        models.save_model(sdn_model, each_network, sdn_params, storedir, epoch=0)

    # run training
    train(networks, storedir, sdn=sdn, device=device)
    # done.

def adv_train_sdns( \
    networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=True, device='cpu'):
    # training strategies
    load_epoch = -1     # IC-only

    # loop over the networks, and set the training configurations
    for each_network in networks:
        cnn_to_tune = re.sub("sdn$", "cnn", each_network)

        # Added by ionmodo
        # because of the above line, the dictionary containing hyperparameters of the CNN will contain
        # the parameter called 'doNormalization' with value False set in create_vgg16bn in network_architectures
        sdn_params = models.load_params(storedir, each_network)
        sdn_params = models.load_cnn_parameters(sdn_params['task'], sdn_params['network_type'])
        sdn_model, _ = utils.cnn_to_sdn(storedir, cnn_to_tune, sdn_params, load_epoch)
        models.save_model(sdn_model, each_network, sdn_params, storedir, epoch=0)

    # do adv-train of an SDN
    adv_train(networks, storedir, \
        attack, max_iter, epsilon, eps_step, sdn=sdn, device=device)
    # done.

def adv_train_ours( \
    networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=True, device='cpu'):
    # training strategies
    load_epoch = -1     # load the trained SDN model

    # loop over the networks, and set the training configurations
    for nidx in range(len(networks)):
        sdn_to_tune = re.sub("sdn$", "ours", networks[nidx])

        # : load the network and parameters
        sdn_params   = models.load_params(storedir, networks[nidx])
        sdn_model, _ = models.load_model(storedir, networks[nidx], epoch=load_epoch)
        models.save_model(sdn_model, sdn_to_tune, sdn_params, storedir, epoch=0)
        print (' : load the [{}] from [{}] ({})'.format( \
            sdn_to_tune, networks[nidx], 'trained' if load_epoch < 0 else 'scratch'))

        # : substitute the network name
        networks[nidx] = sdn_to_tune

    # do adv-train of an SDN
    adv_train(networks, storedir, \
        attack, max_iter, epsilon, eps_step, sdn=sdn, device=device)
    # done.

def adv_train_mixs( \
    networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=True, device='cpu'):
    # training strategies
    load_epoch = -1     # load the trained SDN model

    # loop over the networks, and set the training configurations
    for nidx in range(len(networks)):
        sdn_to_tune = re.sub("sdn$", "ours", networks[nidx])

        # : load the network and parameters
        sdn_params   = models.load_params(storedir, networks[nidx])
        sdn_model, _ = models.load_model(storedir, networks[nidx], epoch=load_epoch)
        models.save_model(sdn_model, sdn_to_tune, sdn_params, storedir, epoch=0)
        print (' : load the [{}] from [{}] ({})'.format( \
            sdn_to_tune, networks[nidx], 'trained' if load_epoch < 0 else 'scratch'))

        # : substitute the network name
        networks[nidx] = sdn_to_tune

    # do adv-train of an SDN
    adv_train(networks, storedir, \
        attack, max_iter, epsilon, eps_step, sdn=sdn, device=device)
    # done.

def train_model( \
    dataset, netname, storedir, \
    cnn=True, cnn_adv=False, sdn=True, sdn_adv=False, \
    attack='ours', max_iter=10, epsilon=8, eps_step=2, device='cpu'):
    cnns = []
    sdns = []

    # set the task to run
    if netname == 'vgg16bn':
        utils.extend_lists(cnns, sdns, \
            models.create_vgg16bn_univ( \
                dataset, storedir, cnn_adv, sdn_adv, \
                attack, max_iter, epsilon, eps_step, 'cs'))
    elif netname == 'resnet56':
        utils.extend_lists(cnns, sdns, \
            models.create_resnet56_univ( \
                dataset, storedir, cnn_adv, sdn_adv, \
                attack, max_iter, epsilon, eps_step, 'cs'))
    elif netname == 'mobilenet':
        utils.extend_lists(cnns, sdns, \
            models.create_mobilenet_univ( \
                dataset, storedir, cnn_adv, sdn_adv, \
                attack, max_iter, epsilon, eps_step, 'cs'))
    else:
        assert False, ('[Train] error: undefined network - {}'.format(netname))


    # train the base models
    if cnn:
        if cnn_adv:
            adv_train(cnns, storedir, \
                'PGD', max_iter, epsilon, eps_step, sdn=False, device=device)
            print ('[Train] trained the base model with PGD attack')
        else:
            train(cnns, storedir, sdn=False, device=device)
            print ('[Train] trained the base model')
    else:
        print ('[Train] we skip the training of a base model')

    # train sdns (IC-only)
    if sdn:
        if sdn_adv:
            adv_train_sdns( \
                sdns, storedir, attack, \
                max_iter, epsilon, eps_step, sdn=True, device=device)
            print ('[Train] trained the SDNs with {} attack'.format(attack))
        else:
            train_sdns(sdns, storedir, sdn=True, device=device)
            print ('[Train] trained the SDNs')
    else:
        print ('[Train] we don\'t train the SDNs, stop.')
    # done.


"""
    Main (for training)
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Train multi-exit networks.')

    # dataset and network
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of the dataset (cifar10 or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='name of the network (vgg16bn, resnet56, or mobilenet)')

    # network training configurations
    parser.add_argument('--cnn', action='store_true',
                        help='train the base CNN (default: False)')
    parser.add_argument('--cnn-adv', action='store_true',
                        help='train the base CNN with adv-training (default: False)')
    parser.add_argument('--sdn', action='store_true',
                        help='train the multi-exit networks with IC-only (default: False)')
    parser.add_argument('--sdn-adv', action='store_true',
                        help='train the multi-exit networks with IC-only + adv-training (default: False)')

    # adversarial training configurations
    parser.add_argument('--attacks', type=str, default='mixs',
                        help='the attack that this script will use for AT (PGD, PGD-avg, PGD-max, ours)')
    parser.add_argument('--maxiter', type=int, default=10,
                        help='maximum number of iterations for the attacks (default: 10)')
    parser.add_argument('--epsilon', type=int, default=8,
                        help='maximum pixel changes of the attacks (default: 8 - pixel)')
    parser.add_argument('--epsstep', type=int, default=2,
                        help='the step size of the perturbations (default: 2 - pixel)')

    # execution parameters
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))

    # run the analysis
    use_device = utils.available_device()
    print ('[Train] use the device: {}'.format(use_device))

    # set the random seed
    random_seed = utils.set_random_seed()
    print ('[Train] set the random seed to: {}'.format(random_seed))

    # set the store location
    model_stores = os.path.join('models', args.dataset)
    utils.create_folder(model_stores)
    print ('[Train] a model will be stored to: {}'.format(model_stores))

    # set the logging folder
    output_folder = 'outputs'
    output_stores = os.path.join(output_folder, \
        '{}_{}_{}_{}_'.format( \
            args.dataset, args.network, \
            'adv' if args.cnn_adv else 'none', \
            'adv' if args.sdn_adv else 'none'))
    if (args.cnn_adv or args.sdn_adv):
        output_stores += '{}_{}_{}_{}'.format( \
            args.attacks, args.maxiter, args.epsilon, args.epsstep)
    utils.create_folder(output_folder)
    utils.start_logger(output_stores)
    print ('[Train] outputs are written down to: {}'.format(output_stores))

    # train a model
    train_model(args.dataset, args.network, model_stores, \
        cnn=args.cnn, cnn_adv=args.cnn_adv, sdn=args.sdn, sdn_adv=args.sdn_adv, \
        attack=args.attacks, max_iter=args.maxiter, epsilon=args.epsilon, \
        eps_step=args.epsstep, device=use_device)
    print ('[Train] done.')
    # done.
