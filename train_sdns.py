"""
    (Adversarially) Train multi-exit architectures
"""
import os, json, sys
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
def train(networks, storedir, sdn=False, ic_only=False, custom_train_loader=None, device='cpu'):
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
        if ic_only:
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
            if ic_only:
                optimizer, scheduler = utils.load_sdn_ic_only_optimizer(network, optimization_params, lr_schedule_params)
                network_name = each_network+'_ic_only'
            else:
                optimizer, scheduler = utils.load_optimizer(network, optimization_params, lr_schedule_params)
                network_name = each_network+'_sdn_training'
        else:
            optimizer, scheduler = utils.load_optimizer(network, optimization_params, lr_schedule_params)
            network_name = each_network

        # FIXME - custom dataset setting, will be moved later on
        # if custom_train_loader is None:
        #     ds = dataset
        # else:
        #     class CustomDataset:
        #         aug_train_loader = None
        #         train_loader = None
        #         test_loader = None
        #
        #     ds = CustomDataset()
        #     if isinstance(custom_train_loader, tuple):
        #         print('Custom train and test loaders')
        #         ds.aug_train_loader = custom_train_loader[0]
        #         ds.train_loader = custom_train_loader[0]
        #         ds.test_loader = custom_train_loader[1]
        #     else:
        #         print('Custom train loader')
        #         ds.aug_train_loader = custom_train_loader[0]
        #         ds.train_loader = custom_train_loader[0]
        #         ds.test_loader = dataset.test_loader

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

def adv_train(networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=False, ic_only=False, device='cpu'):
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
        if ic_only:
            learning_rate = parameters['ic_only']['learning_rate']
            num_epochs    = parameters['ic_only']['epochs']
            milestones    = parameters['ic_only']['milestones']
            gammas        = parameters['ic_only']['gammas']
            parameters['optimizer'] = 'Adam'

            # :: IC-only flag to the model
            network.ic_only = True
        else:
            network.ic_only = False

        # : set the optimizer parameters
        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        # : load the optimizers
        if sdn:
            # :: SDN training (IC-only or training from scratch)
            if ic_only:
                optimizer, scheduler = utils.load_sdn_ic_only_optimizer(network, optimization_params, lr_schedule_params)
                network_name = each_network+'_ic_only'
            else:
                optimizer, scheduler = utils.load_optimizer(network, optimization_params, lr_schedule_params)
                network_name = each_network+'_sdn_training'

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

def train_sdns(networks, storedir, sdn=True, ic_only=False, custom_train_loader=None, device='cpu'):
    # training strategies
    load_epoch = 0
    if ic_only: load_epoch = -1

    # loop over the networks, and set the training configurations
    for each_network in networks:
        cnn_to_tune = each_network.replace('sdn', 'cnn')

        # Added by ionmodo
        # because of the above line, the dictionary containing hyperparameters of the CNN will contain
        # the parameter called 'doNormalization' with value False set in create_vgg16bn in network_architectures
        sdn_params = models.load_params(storedir, each_network)
        sdn_params = models.load_cnn_parameters(sdn_params['task'], sdn_params['network_type'])
        sdn_model, _ = utils.cnn_to_sdn(storedir, cnn_to_tune, sdn_params, load_epoch)
        models.save_model(sdn_model, each_network, sdn_params, storedir, epoch=0)

    # run training
    train(networks, storedir, sdn=sdn, ic_only=ic_only, custom_train_loader=custom_train_loader, device=device)
    # done.

def adv_train_sdns( \
    networks, storedir, \
    attack, max_iter, epsilon, eps_step, sdn=True, ic_only=False, device='cpu'):
    # training strategies
    load_epoch = 0
    if ic_only: load_epoch = -1

    # loop over the networks, and set the training configurations
    for each_network in networks:
        cnn_to_tune = each_network.replace('sdn', 'cnn')

        # Added by ionmodo
        # because of the above line, the dictionary containing hyperparameters of the CNN will contain
        # the parameter called 'doNormalization' with value False set in create_vgg16bn in network_architectures
        sdn_params = models.load_params(storedir, each_network)
        sdn_params = models.load_cnn_parameters(sdn_params['task'], sdn_params['network_type'])
        sdn_model, _ = utils.cnn_to_sdn(storedir, cnn_to_tune, sdn_params, load_epoch)
        models.save_model(sdn_model, each_network, sdn_params, storedir, epoch=0)

    # do adv-train of an SDN
    adv_train(networks, storedir, \
        attack, max_iter, epsilon, eps_step, \
        sdn=sdn, ic_only=ic_only, device=device)
    # done.


def train_model(dataset, netname, storedir, vanilla=False, ic_only=True, device='cpu'):
    cnns = []
    sdns = []

    # set the task to run
    if netname == 'vgg16bn':
        utils.extend_lists(cnns, sdns, models.create_vgg16bn(dataset, storedir, 'cs'))
    elif netname == 'resnet56':
        utils.extend_lists(cnns, sdns, models.create_resnet56(dataset, storedir, 'cs'))
    elif netname == 'mobilenet':
        utils.extend_lists(cnns, sdns, models.create_mobilenet(dataset, storedir, 'cs'))
    else:
        assert False, ('[Train] error: undefined network - {}'.format(netname))

    # train the vanilla models
    if vanilla:
        train(cnns, storedir, sdn=False, device=device)
        print ('[Train] trained the vanilla model (no SDNs)')

    # train sdns (ic-only)
    train_sdns(sdns, storedir, sdn=True, ic_only=ic_only, device=device)
    print ('[Train] trained the SDNs for {}'.format(netname))
    # done.

def advtrain_model( \
    dataset, netname, storedir, \
    attack='', max_iter=10, epsilon=8, eps_step=2, \
    vanilla=True, ic_only=True, device='cpu'):
    cnns = []
    sdns = []

    # set the task to run (adversarial training)
    if netname == 'vgg16bn':
        utils.extend_lists(cnns, sdns, models.create_vgg16bn_adv( \
            dataset, storedir, True, attack, max_iter, epsilon, eps_step, 'cs'))
    elif netname == 'resnet56':
        utils.extend_lists(cnns, sdns, models.create_resnet56_adv( \
            dataset, storedir, True, attack, max_iter, epsilon, eps_step, 'cs'))
    elif netname == 'mobilenet':
        utils.extend_lists(cnns, sdns, models.create_mobilenet_adv( \
            dataset, storedir, True, attack, max_iter, epsilon, eps_step, 'cs'))
    else:
        assert False, ('Error: undefined network - {}'.format(netname))

    # train the vanilla models
    if vanilla:
        adv_train(cnns, storedir, \
            attack, max_iter, epsilon, eps_step, sdn=False, device=device)
        print ('[Adv-Train] trained the vanilla model (no SDNs) with [] attack'.format(attack))

    # train sdns (ic-only)
    adv_train_sdns(sdns, storedir, \
        attack, max_iter, epsilon, eps_step, sdn=True, ic_only=ic_only, device=device)
    print ('[Adv-Train] trained the SDNs for {}'.format(netname))
    # done.


def train_cnns_sdns_w_custom_loaders(dataset, model_path, device='cpu'):

    ic_only = True

    datasets = scenario_1_split()
    for dataset_perc in datasets:

        cnns = []
        sdns = []

        datasets[dataset_perc].add_cifar10_transforms()

        print('Scenario 1 -- Perc: {}'.format(dataset_perc))
        path = os.path.join('models', dataset, 'scenario_1', 'perc_{}'.format(dataset_perc))
        af.create_path(path)

        custom_train_loader = af.ManualData.get_loader(datasets[dataset_perc], shuffle=True)
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(path, dataset, save_type='cd'))
        print(cnns)
        print(sdns)

        if os.path.isfile(os.path.join(path, 'cifar10_vgg16bn_cnn', 'last')):
            print('cont - cnn')
        else:
            train(path, cnns, sdn=False, custom_train_loader=custom_train_loader, device=device)

        if os.path.isfile(os.path.join(path, 'cifar10_vgg16bn_sdn_ic_only', 'last')):
            print('cont - sdn')
        else:
            train_sdns(path, sdns, ic_only=ic_only, custom_train_loader=custom_train_loader, device=device)


"""
    Main (for training)
"""
if __name__ == '__main__':
    print('---------------------------------------')
    print('Date and time:', datetime.now())
    print('Program arguments:', ' '.join(sys.argv))

    parser = argparse.ArgumentParser( \
        description='Train SDN networks.')

    # training configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of the dataset (cifar10 or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='name of the network (vgg16bn, resnet56, or mobilenet)')
    parser.add_argument('--vanilla', action='store_true',
                        help='train the vanilla CNN (default: False)')
    parser.add_argument('--ic-only', action='store_true',
                        help='train the multi-exit networks with IC-only (default: False)')

    # adversarial training configurations
    parser.add_argument('--adv-run', action='store_true',
                        help='train the multi-exit networks with adversarial training (default: False)')
    parser.add_argument('--attacks', type=str, default='ours',
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

    if not args.ic_only:
        print('Error: we do not currently support training SDN from scratch, please refer this functionality in the repository https://github.com/yigitcankaya/Shallow-Deep-Networks')
        exit()

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
        '{}_{}_{}_{}_'.format(args.dataset, args.network, args.vanilla, args.ic_only))
    if args.adv_run: output_stores += '{}_{}_{}_{}'.format(args.attacks, args.maxiter, args.epsilon, args.epsstep)
    utils.create_folder(output_folder)
    utils.start_logger(output_stores)
    print ('[Train] outputs are written down to: {}'.format(output_stores))

    # train a model
    if not args.adv_run:
        train_model(args.dataset, args.network, model_stores, \
            vanilla=args.vanilla, ic_only=args.ic_only, device=use_device)
        print ('[Train] done, training a vanilla model')
    else:
        advtrain_model(args.dataset, args.network, model_stores, \
            attack=args.attacks, max_iter=args.maxiter, epsilon=args.epsilon, eps_step=args.epsstep, \
            vanilla=args.vanilla, ic_only=args.ic_only, device=use_device)
        print ('[Train] done, training an AT model')

    print('Date and time:', datetime.now())
    print('Program arguments:', ' '.join(sys.argv))
    print('---------------------------------------')
    # done.
