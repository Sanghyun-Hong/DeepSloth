"""
    Run attacks on the trained networks
"""
import os, json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# custom libs
import models, utils
from datasets import NumpyDataset, TensorDataset, \
    load_train_loader, load_valid_loader

from attacks.PGDs import PGD, PGD_avg, PGD_max
from attacks.UAP import UAP
import attacks.ours_l1 as ours_l1
import attacks.ours_l2 as ours_l2
import attacks.ours_linf as ours_linf

from delay_attack_cost import compute_delay_metric_w_loader


# ------------------------------------------------------------------------------
#   Sample indexes for UAP attacks and the functions for choosing them
# ------------------------------------------------------------------------------
_samples = [2611, 1566, 8878, 8588, 7818, 6207, 2515, 2019, 3524, 3504,
            6698, 1842,  362, 8877, 6771, 7089, 3722, 9944, 4463, 3451,
             662, 2232, 9104,  178, 7246, 2120, 1008, 1286, 4141, 2807,
            2109, 8774, 4417, 6928, 6512,  644, 3986, 7407, 9924, 7192,
            8399,   83, 5079, 1318,    1, 3158, 8013,  117, 1146,  828,
            1943, 7107, 8928, 5633, 5018, 7592, 5297, 9353, 4675, 7338,
            9670, 6406, 8098, 7567, 5073, 4598, 6368, 9541, 7147, 2187,
            5062, 7053, 4988, 1438, 4807, 9825, 8650, 3218, 7979, 6640,
            4344,  233, 2855, 9624, 9437, 5624, 5007, 8015, 9309, 6603,
            8685, 4304,  257, 4533, 7598, 6337, 1007, 4956, 5053, 5571]

def _compose_samples(dataloader, indexes):
    sample_data   = []
    sample_labels = []

    # loop over the loader
    for didx, (data, labels) in enumerate( \
        tqdm(dataloader, desc='[Run attacks:compose-samples]')):
        if (indexes is not None) and (didx not in indexes): continue
        sample_data.append(data.clone())
        sample_labels.append(labels.clone())
    # end for didx....

    # convert to the tensor
    sample_data   = torch.cat(sample_data, dim=0)
    sample_labels = torch.cat(sample_labels, dim=0)
    return sample_data, sample_labels


# ------------------------------------------------------------------------------
#   Main attack code: Run or analysis
# ------------------------------------------------------------------------------
def run_attack(args, use_cuda=False):

    # load the clean validation set
    train_loader = load_train_loader(args.dataset, nbatch=args.batch_size)
    valid_loader = load_valid_loader(args.dataset, nbatch=args.batch_size)
    print ('[Run attack] load the valid set - {}'.format(args.dataset))

    # choose the 100 samples in our interest
    if 'UAP' == args.attacks:
        # : sanity check
        if 100 != args.nsample:
            assert False, ('Error: unsupported # samples - {}'.format(args.nsample))

        # : choose the samples
        chosen_data, chosen_labels = \
            _compose_samples(valid_loader, _samples)
    # end if 'UAP'...

    # load the network
    net_cnn = False
    netpath = os.path.join('models', '{}'.format(args.dataset))
    netname = '{}_{}_{}_{}_{}_10_8_2_{}'.format( \
        args.dataset, args.network, \
        'none' if not args.cnn_adv else 'adv', \
        'none' if not args.sdn_adv else 'adv', \
        args.net_adv, args.nettype)
    model, params = models.load_model(netpath, netname, epoch='last')
    if use_cuda: model.cuda()
    model.eval()
    print ('[Run attack] load the model [{}], from [{}]'.format(netname, netpath))


    """
        Perform attacks (l1, l2, PGDs, Ours, UAP)
    """
    if 'PGD' in args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # dataset holder
        total_adv_data   = []
        total_adv_labels = []

        # run the adversarial attacks,
        #   and store them for the analysis
        for bidx, (data, labels) in enumerate( \
            tqdm(valid_loader, desc='[Run attack]')):
            if use_cuda:
                data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels, requires_grad=False)

            # : conduct the attacks
            if 'PGD' == args.attacks:
                data_adv = PGD( \
                    data, None, F.cross_entropy, \
                    y=labels, model=model, \
                    eps=8/255., steps=args.maxiter, \
                    gamma=2/255., norm=args.ellnorm, \
                    randinit=True, cnn=net_cnn, cuda=use_cuda).data
                data_adv = Variable(data_adv, requires_grad=False)

            elif 'PGD-avg' == args.attacks:
                data_adv = PGD_avg( \
                    data, None, F.cross_entropy, \
                    y=labels, model=model, \
                    eps=8/255., steps=args.maxiter, \
                    gamma=2/255., norm=args.ellnorm, \
                    randinit=True, cnn=net_cnn, cuda=use_cuda).data
                data_adv = Variable(data_adv, requires_grad=False)

            elif 'PGD-max' == args.attacks:
                data_adv = PGD_max( \
                    data, None, F.cross_entropy, \
                    y=labels, model=model, \
                    eps=8/255., steps=args.maxiter, \
                    gamma=2/255., norm=args.ellnorm, \
                    randinit=True, cnn=net_cnn, cuda=use_cuda).data
                data_adv = Variable(data_adv, requires_grad=False)

            # : [DEBUG] compute output and check if the attack was successful
            # org_outputs = sdn_model(data)
            # adv_outputs = sdn_model(data_adv)
            # for oidx in range(len(org_outputs)):
            #     each_org = org_outputs[oidx]
            #     each_adv = adv_outputs[oidx]
            #     print (' : {}/{} [label: {}]'.format( \
            #         torch.argmax(each_org).item(), \
            #         torch.argmax(each_adv).item(), labels[0]))
            # exit()

            # : [DEBUG] to show the crafted samples
            # import torchvision.utils as vutils
            # vutils.save_image(data_adv, '{}_adv_samples.png'.format(args.attacks))
            # exit()

            # : save the adversarial sample to an array
            total_adv_data.append(data_adv)
            total_adv_labels.append(labels)

        # end for bidx...

        # concatenate the entire results
        total_adv_data   = torch.cat(total_adv_data, dim=0).cpu().numpy()
        total_adv_labels = torch.cat(total_adv_labels, dim=0).cpu().numpy()
        print ('[Run attack] create adv. samples with [{}], [{}] samples'.format(args.attacks, len(total_adv_data)))

        # store them to a file
        with open(os.path.join(save_folder, '{}_{}_samples.pickle'.format(args.attacks, args.ellnorm)), 'wb') as handle:
            pickle.dump((total_adv_data, total_adv_labels), handle, protocol=4)

        exit()

    elif 'UAP' == args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # run the UAP adversarial attacks,
        #   and store them for the analysis
        uap_data = UAP(chosen_data, chosen_labels, model, data_shape=(1, 3, 32, 32), \
            max_uiter=100, max_diter=10, max_norm=2/255., cuda=use_cuda)
        print ('[Run attack] UAP with [{}] samples'.format(args.nsample))

        # store the perturbation to a file
        with open(os.path.join(save_folder, '{}_perturbations.pickle'.format(args.attacks)), 'wb') as handle:
            pickle.dump(uap_data.numpy(), handle, protocol=4)

        exit()

    elif 'ours' == args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        # run the DeepSloth + universal DeepSloth,
        #   and store them for the analysis
        if 'linf' == args.ellnorm:
            total_adv_data, total_adv_labels = \
                ours_linf.craft_per_sample_perturb_attack( \
                    model, valid_loader, device='cuda' if use_cuda else 'cpu')

        else:
            assert False, ('Error: unsupported norm - {}'.format(args.ellnorm))

        """
            Take the max. iterations, since the attack is done
              with K (any number) iterations and save per K/10 iterations
        """
        with open(os.path.join(save_folder, '{}_{}_clean.pickle'.format(args.attacks, args.ellnorm)), 'wb') as handle:
            pickle.dump((total_adv_data[0], total_adv_labels), handle, protocol=4)

        with open(os.path.join(save_folder, '{}_{}_persample.pickle'.format(args.attacks, args.ellnorm)), 'wb') as handle:
            pickle.dump((total_adv_data[-1], total_adv_labels), handle, protocol=4)

        # stop at here...
        exit()

    else:
        assert False, ('Error: unsupported attack - {}'.format(_attacks))
    # done.

def run_analysis(args, use_cuda=False):

    # load the network
    net_cnn = False
    netpath = os.path.join('models', '{}'.format(args.dataset))
    netname = '{}_{}_{}_{}_{}_10_8_2_{}'.format( \
        args.dataset, args.network, \
        'none' if not args.cnn_adv else 'adv', \
        'none' if not args.sdn_adv else 'adv', \
        args.net_adv, args.nettype)
    model, params = models.load_model(netpath, netname, epoch='last')
    if use_cuda: model.cuda()
    model.eval()
    print ('[Run analysis] load the model [{}], from [{}]'.format(netname, netpath))


    """
        Perform analysis (PGDs, Our, or UAP)
    """
    if 'PGD' in args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        analyze_dir = os.path.join( \
            'analysis', args.dataset, netname)

        # create dir.
        if not os.path.exists(analyze_dir): os.makedirs(analyze_dir)
        print ('[Run analysis] create an analysis folder [{}]'.format(analyze_dir))

        # test configure
        datafiles = [
            os.path.join(save_folder, \
                '{}_{}_samples.pickle'.format(args.attacks, args.ellnorm))
        ]
        rad_limits = [5]

        # check the outputs
        for eachrad in rad_limits:
            for eachfile in datafiles:
                print ('--------')
                with open(eachfile, 'rb') as handle:
                    attack_data, attack_labels = pickle.load(handle)

                # > save some samples
                samples_fname = os.path.join(analyze_dir, \
                    '{}_samples.png'.format(eachfile.split('/')[-1].replace('.pickle', '')))
                samples_size  = 8
                samples_data  = torch.from_numpy(attack_data[:samples_size])
                vutils.save_image(samples_data, samples_fname)

                # > compose dataset
                delayed_dataset= TensorDataset(attack_data, attack_labels)
                advdata_loader = DataLoader( \
                    delayed_dataset, shuffle=False, batch_size=1)
                print(f'[{args.dataset}][{eachfile}] SDN evaluations')

                # > analyze
                analysis_file = os.path.join(analyze_dir, \
                    '{}_{}_analysis'.format(eachfile.split('/')[-1].replace('.pickle', ''), eachrad))
                plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                    compute_delay_metric_w_loader( \
                        'models/{}', args.dataset, args.network, \
                        eachrad, advdata_loader, analysis_file, \
                        adv=True, \
                        cnnadv=args.cnn_adv, sdnadv=args.sdn_adv, \
                        netadv=args.net_adv, nettype=args.nettype)
                # print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

        print ('--------')
        print ('[Run analysis] Done.'); exit()
        # stop here...

    elif 'UAP' in args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        analyze_dir = os.path.join( \
            'analysis', args.dataset, netname)

        # create dir.
        if not os.path.exists(analyze_dir): os.makedirs(analyze_dir)
        print ('[Run analysis] create an analysis folder [{}]'.format(analyze_dir))

        # test configure
        datafiles = [
            os.path.join(save_folder, \
                '{}_perturbations.pickle'.format(args.attacks))
        ]
        rad_limits = [5]

        # : load the validation dataset
        valid_loader = load_valid_loader(args.dataset)
        print ('[Run analysis] load the valid set - {}'.format(args.dataset))

        # check the outputs
        for eachrad in rad_limits:
            for eachfile in datafiles:
                print ('--------')

                # > load the perturbation
                with open(eachfile, 'rb') as handle:
                    perturb = pickle.load(handle)
                attack_data, attack_labels = \
                    ours_linf.apply_perturb_attack(valid_loader, perturb)

                # > save some samples
                samples_fname = os.path.join(analyze_dir, \
                    '{}_samples.png'.format(eachfile.split('/')[-1].replace('.pickle', '')))
                samples_size  = 8
                samples_data  = torch.from_numpy(attack_data[:samples_size])
                vutils.save_image(samples_data, samples_fname)

                # > compose dataset
                delayed_dataset= TensorDataset(attack_data, attack_labels)
                advdata_loader = DataLoader( \
                    delayed_dataset, shuffle=False, batch_size=1)
                print(f'[{args.dataset}][{eachfile}] SDN evaluations')

                # > analyze
                analysis_file = os.path.join(analyze_dir, \
                    '{}_{}_analysis'.format(eachfile.split('/')[-1].replace('.pickle', ''), eachrad))
                plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                    compute_delay_metric_w_loader( \
                        'models/{}', args.dataset, args.network, \
                        eachrad, advdata_loader, analysis_file, \
                        adv=True, \
                        cnnadv=args.cnn_adv, sdnadv=args.sdn_adv, \
                        netadv=args.net_adv, nettype=args.nettype)
                # print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

        print ('--------')
        print ('[Run analysis] Done.'); exit()
        # stop here...

    elif 'ours' == args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, netname)
        analyze_dir = os.path.join( \
            'analysis', args.dataset, netname)

        # create dir.
        if not os.path.exists(analyze_dir): os.makedirs(analyze_dir)
        print ('[Run analysis] create an analysis folder [{}]'.format(analyze_dir))

        # test configure
        datafiles = [
            os.path.join(save_folder, \
                '{}_{}_{}.pickle'.format( \
                    args.attacks, args.ellnorm, suffix)) \
            for suffix in ['clean', 'persample']
        ]
        rad_limits = [5]

        # check the outputs
        for eachrad in rad_limits:
            for eachfile in datafiles:
                print ('--------')
                with open(eachfile, 'rb') as handle:
                    attack_data, attack_labels = pickle.load(handle)

                # > save some samples
                samples_fname = os.path.join(analyze_dir, \
                    '{}_samples.png'.format(eachfile.split('/')[-1].replace('.pickle', '')))
                samples_size  = 8
                samples_data  = torch.from_numpy(attack_data[:samples_size])
                vutils.save_image(samples_data, samples_fname)

                # > compose dataset
                delayed_dataset= TensorDataset(attack_data, attack_labels)
                advdata_loader = DataLoader( \
                    delayed_dataset, shuffle=False, batch_size=1)
                print(f'[{args.dataset}][{eachfile}] SDN evaluations')

                # > analyze
                analysis_file = os.path.join(analyze_dir, \
                    '{}_{}_analysis'.format(eachfile.split('/')[-1].replace('.pickle', ''), eachrad))
                plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                    compute_delay_metric_w_loader( \
                        'models/{}', args.dataset, args.network, \
                        eachrad, advdata_loader, analysis_file, \
                        adv=True, \
                        cnnadv=args.cnn_adv, sdnadv=args.sdn_adv, \
                        netadv=args.net_adv, nettype=args.nettype)
                # print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

        print ('--------')
        print ('[Run analysis] Done.'); exit()
        # stop here...

    else:
        assert False, ('Error: unsupported attacks - {}'.format(args.attacks))
    # done.


"""
    Main (Run the PGD/UAP/our attacks)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
        description='Run the PGD/UAP/our attacks on our adaptive AT nets.')

    # attack modes
    parser.add_argument('--runmode', type=str, default='attack',
                        help='runmode of the script (attack, or analysis)')

    # basic configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of the dataset (cifar10 or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='location of the network (vgg16bn, resnet56, or mobilenet)')
    parser.add_argument('--net-adv', type=str, default='PGD',
                        help='the type of the network that is trained with (PGD, PGD-avg, PGD-max, ours)')
    parser.add_argument('--nettype', type=str, default='sdn',
                        help='the type of the network that we use (cnn, sdn, or ours)')
    parser.add_argument('--cnn-adv', action='store_true',
                        help='train the base CNN with adv-training (default: False)')
    parser.add_argument('--sdn-adv', action='store_true',
                        help='train the multi-exit networks with IC-only + adv-training (default: False)')

    # attack configurations
    parser.add_argument('--attacks', type=str, default='PGD',
                        help='the attack that this script will use (PGD, PGD-avg, PGD-max, UAP, ours)')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='maximum number of iterations for the attacks (default: 10 / 1 for universal)')
    parser.add_argument('--ellnorm', type=str, default='linf',
                        help='the norm used to bound the attack (default: linf - l1 and l2)')
    parser.add_argument('--nsample', type=int, default=100,
                        help='the number of samples consider (for UAP)')

    # hyper-parameters
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='the batch size used to craft adv. samples (default: 1000)')

    # execution parameters
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))

    # set cuda if available
    set_cuda = True if 'cuda' == utils.available_device() else False
    print ('[{}] set cuda [{}]'.format(set_cuda, args.runmode))

    # run the attack or analysis
    if 'attack' == args.runmode:
        run_attack(args, use_cuda=set_cuda)
    elif 'analysis' == args.runmode:
        run_analysis(args, use_cuda=set_cuda)
    else:
        assert False, ('Error: undefined run-mode - {}'.format(args.runmode))
    # done.
