"""
    Script for computing the DeepSloth costs
"""
import os, copy
import random
import pickle
import numpy as np

# torch and stuffs
import torch
import matplotlib.pyplot as plt

# suppress warnings (arctanh)
import warnings
warnings.filterwarnings("ignore")

# custom libs
import models, utils
import model_funcs as mf
import attacks.ours_linf as DeepSloth
from profiler import profile_sdn


# ------------------------------------------------------------------------------
#   Plotting codes...
# ------------------------------------------------------------------------------
def draw_plot(plots_data, accs, latenesses, labels, title, save_path):
    fig = plt.figure()
    ax = plt.axes()

    for plot_idx, plot_data in enumerate(plots_data):
        legend_label = f'{labels[plot_idx]}(ACC:{accs[plot_idx]:.1f}LTNS:{latenesses[plot_idx]:.2f})'
        ax.plot(*plot_data, label=legend_label, linewidth=3.0)

    ax.set_xlim(-0.1, 1.1)
    plt.title(title, fontsize='small')
    plt.ylabel('Frac. Instances That Exit (Cumulative)')
    plt.xlabel('Frac. Comp. Cost Over the Full Network')
    plt.grid(True)

    plt.legend(loc='best')
    plt.savefig(save_path)
    # done.

def convert_set_of_early_exit_samples_to_cumulative_dist(ic_exits, total_samples):
    layers = sorted(list(ic_exits.keys()))

    cum_correct = set()

    layer_cumul_dist = [0]

    for layer in layers:
        cur_correct = ic_exits[layer]
        cum_correct = cum_correct | cur_correct
        layer_cumul_dist.append(len(cum_correct))

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist

def convert_num_early_exits_at_each_ic_to_cumulative_dis(ic_exits, total_samples):
    num_exits = len(ic_exits)

    layer_cumul_dist = [0]

    running_total = 0
    for cur_exit in range(num_exits):
        running_total += ic_exits[cur_exit]
        layer_cumul_dist.append(running_total)

    layer_cumul_dist[-1] = total_samples
    layer_cumul_dist = [val / total_samples for val in layer_cumul_dist]
    return layer_cumul_dist


# ic_exits  --- at each IC, the samples that exit (list-dict of set)
# ic_costs --- the output of the profiler_sdn for the network (dict)
# total samples --- how many samples were in the test set (int)
# return --> the data to draw a delay plot and the area under the curve as our delay metric
def get_plot_data_and_auc(layer_cumul_dist, ic_costs):
    layers = sorted(list(ic_costs.keys()))

    c_i = {layer: ic_costs[layer] / ic_costs[layers[-1]] for layer in layers}
    c_i = [c_i[layer] for layer in layers]
    c_i.insert(0, 0)

    plot_data = [c_i, layer_cumul_dist]

    area_under_curve = np.trapz(layer_cumul_dist, x=c_i)

    return plot_data, area_under_curve


# to test the delay metric and create a simple plot
def get_oracle_latency_plot(path, task, network, device='cpu'):
    models_path = path.format(task)
    sdn_name = task + '_' + network + '_sdn_ic_only'

    save_name = os.path.join(models_path, sdn_name, 'perfect_oracle_exit_rates')
    save_pickle = f'{save_name}.pickle'

    if utils.file_exists(save_pickle):
        print('get_oracle_latency_plot: Results file exists, loading the results from the file...')
        with open(save_pickle, 'rb') as handle:
            results = pickle.load(handle)

        plot_data, early_exit_auc = results['plot_data'], results['auc_delay_metric']
        orig_acc, early_exit_acc = results['orig_acc'], results['early_exit_acc']

    else:
        print('get_oracle_latency_plot: Results does not exits, running the experiment...')

        sdn_model, sdn_params = models.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)
        dataset = utils.load_dataset(sdn_params['task'], doNormalization=sdn_params.get('doNormalization', False))

        top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))

        total_samples = utils.loader_inst_counter(dataset.test_loader)

        layer_correct, _, _, _ = mf.sdn_get_detailed_results(sdn_model, loader=dataset.test_loader, device=device)

        layer_cumul_dist = convert_set_of_early_exit_samples_to_cumulative_dist(layer_correct, total_samples)

        orig_acc = top1_test[-1]

        early_exit_acc = (len(set.union(*list(layer_correct.values()))) / total_samples) * 100

        c_i = profile_sdn(sdn_model, sdn_model.input_size, device)[0]

        layers = sorted(list(c_i.keys()))

        plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, c_i)
        early_exit_auc = (1 - early_exit_auc)

        results = {}
        results['plot_data'] = plot_data
        results['auc_delay_metric'] = early_exit_auc
        results['orig_acc'] = orig_acc
        results['early_exit_acc'] = early_exit_acc
        results['early_exit_counts'] = [len(layer_correct) for layer in layers]
        results['total_samples'] = total_samples

        with open(save_pickle, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(f'Oracle - Orig Acc: {orig_acc:.2f} - Early Exit Acc: {early_exit_acc:.2f} - Early Exit AUC: {early_exit_auc:.2f}')

    return plot_data, early_exit_auc, orig_acc, early_exit_acc


# finds and returns the respective confidence threshold values that satisfy the accuracy drop criteria
def get_rad_confidence_threshold( \
    path, task, network, rad_limit=5, device='cpu', \
    adv=False, cnnadv=False, sdnadv=False, netadv='PGD', nettype='sdn'):
    threshold_stats = test_and_save_threshold_results( \
        path, task, network, device=device, \
        adv=adv, cnnadv=cnnadv, sdnadv=sdnadv, netadv=netadv, nettype=nettype)
    threshold_accs = threshold_stats['threshold_accs']
    threshold_aucs = threshold_stats['threshold_comp_costs']
    orig_acc = threshold_stats['sdn_top1_acc'][-1]  # accuracy of the base cnn
    target_acc = orig_acc * (1 - (rad_limit / 100))

    min_cost = np.inf
    threshold = -1
    max_acc = -1
    for cur_threshold in threshold_accs:
        cur_acc = threshold_accs[cur_threshold]
        cur_cost = threshold_aucs[cur_threshold]
        if cur_acc > target_acc and cur_cost < min_cost and cur_acc > max_acc:
            threshold = cur_threshold
            min_cost = cur_cost
            max_acc = cur_acc

    return threshold, threshold_accs[threshold], threshold_aucs[threshold]


# test the network for different confidence thresholds, save the results and return the threshold values that satisfy the criteria
# it saves the results to the model's parameters file for fast access in the future
def test_and_save_threshold_results( \
    path, task, network, manual_loader=None, device='cpu', \
    adv=False, cnnadv=False, sdnadv=False, netadv='PGD', nettype='sdn'):
    models_path = path.format(task)

    if adv:
        sdn_name = '{}_{}_{}_{}_{}_10_8_2_{}'.format( \
            task, network, \
            'none' if not cnnadv else 'adv', \
            'none' if not sdnadv else 'adv', \
            netadv, nettype)
    else:
        sdn_name = task + '_' + network + '_sdn_ic_only'
    print (' : [test-save] load the model [{}]'.format(sdn_name))

    save_pickle = os.path.join(models_path, sdn_name, 'confidence_threshold_stats')
    save_pickle = f'{save_pickle}.pickle'

    if utils.file_exists(save_pickle):
        # print('test_and_save_threshold_results: Results file exists, loading the results from the file...')
        with open(save_pickle, 'rb') as handle:
            threshold_stats = pickle.load(handle)

        return threshold_stats

    else:
        threshold_stats = {}
        sdn_model, sdn_params = models.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)
        c_i, _ = profile_sdn(sdn_model, sdn_model.input_size, device)


        if manual_loader is not None:
            loader, one_batch_loader = manual_loader
            assert one_batch_loader.batch_size == 1, 'manual loader batch size should be one'
        else:
            dataset = utils.load_dataset(sdn_params['task'], doNormalization=sdn_params.get('doNormalization', False))
            one_batch_dataset = utils.load_dataset(sdn_params['task'], batch_size=1, doNormalization=sdn_params.get('doNormalization', False))
            loader, one_batch_loader = dataset.test_loader, one_batch_dataset.test_loader


        threshold_stats['sdn_top1_acc'], threshold_stats['sdn_top5_acc'] = mf.sdn_test(sdn_model, loader, device)
        print(threshold_stats['sdn_top1_acc'])

        # to test early-exits with the SDN
        total_samples = utils.loader_inst_counter(loader)

        print('test_and_save_threshold_results:Testing with different confidence thresholds...')
        confidence_thresholds = np.linspace(0.05, 0.95, 19)  # search for the confidence threshold for early exits

        sdn_model.forward = sdn_model.early_exit
        sdn_model.output_to_return_when_ICs_are_delayed = 'network_output'

        threshold_comp_costs = {}
        threshold_accs = {}

        for threshold in confidence_thresholds:
            sdn_model.confidence_threshold = threshold

            # change the forward func for sdn to forward with cascade
            top1_test, _, early_exit_counts, _, _ = mf.sdn_test_early_exits(sdn_model, one_batch_loader, device)

            layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts, total_samples)
            _, auc_delay_metric = get_plot_data_and_auc(layer_cumul_dist, c_i)

            threshold_accs[threshold] = top1_test
            threshold_comp_costs[threshold] =  (1 - auc_delay_metric)

            print('Threshold {0:.2f} - ACC: {1:.2f} - LATENESS: {2:.2f}'.format(threshold, top1_test, (1 - auc_delay_metric)))

        threshold_stats['threshold_accs'] = threshold_accs
        threshold_stats['threshold_comp_costs'] = threshold_comp_costs

        with open(save_pickle, 'wb') as f:
            pickle.dump(threshold_stats, f, protocol=4)

    return threshold_stats


# first loads the model then sets the early exit confidence threshold based on the rad limit
# then forward passes the loader (batch size of 1) to test the early exits
# plot save name --- file name of the plot of the early exit rates
# Finrally computes and returns the delay metric (AUC) along with the data required to plot the early exit distribution
# It also returns the accuracy with the early exits
def compute_delay_metric_w_loader( \
    path, task, network, rad_limit, loader, save_name, \
    adv=False, cnnadv=False, sdnadv=False, netadv='PGD', nettype='sdn', \
    device='cpu'):
    save_pickle = f'{save_name}.pickle'

    threshold, orig_acc, orig_auc = \
        get_rad_confidence_threshold( \
            path, task, network, rad_limit, device, \
            adv=adv, cnnadv=cnnadv, sdnadv=sdnadv, netadv=netadv, nettype=nettype)

    if utils.file_exists(save_pickle):
        # print('compute_delay_metric: Results file exists, loading the results from the file...')
        with open(save_pickle, 'rb') as handle:
            results = pickle.load(handle)
            plot_data, early_exit_auc, early_exit_acc = results['plot_data'], results['auc_delay_metric'], results['early_exit_acc']

        # small fix
        early_exit_lateness = (1 - early_exit_auc)

    else:
        # print('compute_delay_metric: Results file does not exist, running the experiment...')

        models_path = path.format(task)

        if adv:
            sdn_name = '{}_{}_{}_{}_{}_10_8_2_{}'.format( \
                task, network, \
                'none' if not cnnadv else 'adv', \
                'none' if not sdnadv else 'adv', \
                netadv, nettype)
        else:
            sdn_name = task + '_' + network + '_sdn_ic_only'
        print (' : [compute-delay] load the model [{}]'.format(sdn_name))

        sdn_model, _ = models.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        c_i = profile_sdn(sdn_model, sdn_model.input_size, device)[0]

        # set the threshold
        sdn_model.confidence_threshold = threshold
        sdn_model.forward = sdn_model.early_exit

        # IMPORTANT: In SDN we take the most confident exit if no exit exceeds the threshold
        # But for this paper let's take the final classifier if that's the case
        sdn_model.output_to_return_when_ICs_are_delayed = 'network_output'

        # loader needs to be batch_size = 1 for this to work!
        total_samples = utils.loader_inst_counter(loader)
        early_exit_acc, _, early_exit_counts, non_early_counts, _ = mf.sdn_test_early_exits(sdn_model, loader, device)

        layer_cumul_dist = convert_num_early_exits_at_each_ic_to_cumulative_dis(early_exit_counts, total_samples)
        print (f' RAD<{rad_limit} (T={threshold:.2f}) Exit counts: {early_exit_counts}')

        plot_data, early_exit_auc = get_plot_data_and_auc(layer_cumul_dist, c_i)

        early_exit_lateness = (1 - early_exit_auc)

        results = {}
        results['plot_data'] = plot_data
        results['auc_delay_metric'] = early_exit_auc
        results['early_exit_acc'] = early_exit_acc
        results['early_exit_counts'] = early_exit_counts
        results['non_early_counts'] = non_early_counts
        results['total_samples'] = total_samples

        with open(save_pickle, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f' RAD<{rad_limit} (T={threshold:.2f}) Orig Acc: {orig_acc:.2f} - Early Exit Acc: {early_exit_acc:.2f} - Orig Efficacy: {(1-orig_auc):.2f} - Early Exit Efficacy: {early_exit_auc:.2f}')
    return plot_data, (1-orig_auc), early_exit_auc, orig_acc, early_exit_acc


# the wb network is to craft the attack, the bb are to transfer
def apply_univ_perturb_attack(path, task, wb_network, bb_networks, rad_limit, class_to_attack=None, device='cpu'):
    results_path = os.path.join('univ_attack_results', task, wb_network)
    utils.create_folder(results_path)

    class_info = 'all' if class_to_attack is None else class_to_attack
    save_pickle = os.path.join(results_path, f'up_perturbs_class_{class_info}.pickle')

    if utils.file_exists(save_pickle):
        print('apply_univ_perturb_attack: Perturbs file exists, loading the results from the file...')
        with open(save_pickle, 'rb') as handle:
            perturbs = pickle.load(handle)
    else:
        print('apply_univ_perturb_attack: Perturbs file does not exist, running the experiment...')

        models_path = path.format(task)
        sdn_name = task + '_' + wb_network + '_sdn_ic_only'
        sdn_model = models.load_model(models_path, sdn_name, epoch=-1)[0].to(device)

        if class_to_attack is not None:
            train_loader = utils.ManualData.get_loader(utils.ManualData(*utils.get_task_class_data(task, get_class=class_to_attack)[:2]), batch_size=128)
        else:
            train_loader = utils.load_dataset(task, batch_size=128, doNormalization=False).train_loader

        perturbs = DeepSloth.craft_universal_perturb_attack(sdn_model, train_loader, device=device)
        with open(save_pickle, 'wb') as f:
            pickle.dump(perturbs, f, pickle.HIGHEST_PROTOCOL)

    if class_to_attack is not None:
        test_loader = utils.ManualData.get_loader(utils.ManualData(*utils.get_task_class_data(task, get_class=class_to_attack)[2:]), batch_size=128)
    else:
        test_loader = utils.load_dataset(task, doNormalization=False).test_loader

    networks = list(set([wb_network]) | set(bb_networks))
    # get_perturbs = [('clean', perturbs[0]), ('noise', perturbs[1]), ('attack', perturbs[-1])]

    get_perturbs = [('attack', perturbs[-1])]
    # networks = [wb_network]

    accs = []
    latenesses = []
    for network in networks:
        plots_data = []
        early_exit_accs = []
        early_exit_latenesses = []
        labels = []
        for perturb_info, perturb in get_perturbs:
            print('\napply_univ_perturb_attack: Network: {} -- ({} on: {})'.format(network, perturb_info, wb_network))
            images_save_path = os.path.join(results_path, f'up_imgs_{perturb_info}_class_{class_info}') if perturb_info != 'clean' else None
            attack_data, attack_labels = DeepSloth.apply_perturb_attack(test_loader, perturb)   # , images_save_path=images_save_path)
            delayed_dataset = utils.ManualData(data=attack_data, labels=attack_labels)
            loader_delayed = utils.ManualData.get_loader(delayed_dataset, batch_size=1)
            save_path = os.path.join(results_path, f'up_results_to_{network}_rad_{rad_limit}_class_{class_info}_perturb_{perturb_info}')
            print (save_path)

            plot, cl_lateness, ee_lateness, cl_acc, ee_acc = \
                compute_delay_metric_w_loader(path, task, network, rad_limit, loader_delayed, save_name=save_path, device=device)
            plots_data.append(plot)
            early_exit_accs.append(ee_acc)
            early_exit_latenesses.append(ee_lateness)
            labels.append(perturb_info)
            accs.append(ee_acc)
            latenesses.append(ee_lateness)

        plot_save_path = os.path.join(results_path, f'up_results_to_{network}_rad_{rad_limit}_class_{class_info}')
        title = f'Univ. ({task}) - RAD<{rad_limit} - {wb_network} to {network} - Class: {class_info}'
        draw_plot(plots_data, early_exit_accs, early_exit_latenesses, labels, title, plot_save_path)

    return accs, latenesses

def apply_persample_perturb_attack(path, task, wb_network, bb_networks, rad_limit, device='cpu'):
    models_path = path.format(task)

    results_path = os.path.join('persample_attack_results', task, wb_network)
    utils.create_folder(results_path)

    save_pickle = os.path.join(results_path, f'attacked_dataset.pickle')

    if utils.file_exists(save_pickle):
        print('apply_persample_perturb_attack: Attack samples file exists, loading the results from the file...\n')
        with open(save_pickle, 'rb') as handle:
            attack_results = pickle.load(handle)
            attack_data_iters = attack_results['attack_data_iters']
            attack_labels = attack_results['attack_labels']

    else:
        print('apply_persample_perturb_attack: Attack samples does not exist, running the experiment...\n')
        models_path = path.format(task)
        sdn_name = task + '_' + wb_network + '_sdn_ic_only'
        sdn_model = models.load_model(models_path, sdn_name, epoch=-1)[0].to(device)
        dataset = utils.load_dataset(task, doNormalization=False)
        test_loader = dataset.test_loader
        attack_data_iters, attack_labels = DeepSloth.craft_per_sample_perturb_attack(sdn_model, test_loader, device)
        attack_results = {}
        attack_results['attack_data_iters'] = attack_data_iters
        attack_results['attack_labels'] = attack_labels

        with open(save_pickle, 'wb') as f:
            pickle.dump(attack_results, f, pickle.HIGHEST_PROTOCOL)

    networks = list(set([wb_network]) | set(bb_networks))

    get_iters = [('clean', attack_data_iters[0]), ('noise', attack_data_iters[1]), ('attack', attack_data_iters[-1])]

    for network in networks:
        plots_data = []
        early_exit_accs = []
        early_exit_latenesses = []
        labels = []
        for data_info, data in get_iters:
            print('\napply_persample_perturb_attack: Attacking: {}, crafted on: {} ({})\n'.format(network, wb_network, data_info))
            delayed_dataset = utils.ManualData(data=data, labels=attack_labels, device=device)
            loader_delayed = utils.ManualData.get_loader(delayed_dataset, batch_size=1, device=device)
            save_path = os.path.join(results_path, f'ps_results_to_{network}_iters_rad_{rad_limit}_data_{data_info}')
            plot, ee_lateness, ee_acc = compute_delay_metric_w_loader(path, task, network, rad_limit, loader_delayed, save_name=save_path, device=device)
            plots_data.append(plot)
            early_exit_accs.append(ee_acc)
            early_exit_latenesses.append(ee_lateness)
            labels.append(data_info)

        plot_save_path = os.path.join(results_path, f'ps_results_to_{network}_rad_{rad_limit}')
        title = f'Per Sample. ({task}) - RAD<{rad_limit} - {wb_network} to {network}'
        draw_plot(plots_data, early_exit_accs, early_exit_latenesses, labels, title, plot_save_path)


"""
    Main (for computing our metrics on networks)
"""
if __name__ == "__main__":
    # to make it reproducible
    utils.set_random_seed()

    # set device
    device = utils.available_device()

    # configurations
    tasks       = ['cifar10']
    networks    = ['vgg16bn']   # , 'resnet56', 'mobilenet']
    rad_limits  = [5]
    path        = 'models/{}'

    num_attack_classes = 10

    for task in tasks:
        attack_classes = random.sample(list(range(utils.get_task_num_classes(task))), num_attack_classes)
        for network in networks:
            print('Task {}, Network: {}'.format(task, network))
            print('\nAttacking all classes')
            # for rad_limit in rad_limits:
            #     print('\nRad limit: {}'.format(rad_limit))
            #     apply_univ_perturb_attack(path, task, network, networks, rad_limit, None, device)

            for rad_limit in rad_limits:
                print('\nRad limit: {}'.format(rad_limit))
                accs = []
                latenesses = []
                for attack_class in attack_classes:
                    print('\nAttacking class: {}'.format(attack_class))
                    acc, lateness = apply_univ_perturb_attack(path, task, network, networks, rad_limit, attack_class, device)
                    latenesses.append(lateness)
                    accs.append(acc)
                print(accs)
                print(latenesses)
                print(f'acc mean: {np.mean(accs):.2f}, acc std:{np.std(accs):.2f}')
                print(f'lateness mean: {np.mean(latenesses):.2f}, lateness std:{np.std(latenesses):.2f}')

            # print('\nApplying per-sample attack')
            # for rad_limit in rad_limits:
            #     print('\nRad limit: {}'.format(rad_limit))
            #     apply_persample_perturb_attack(path, task, network, networks, rad_limit, device)

    ######## BASELINE STUFF ##############
    '''
    results_path = 'oracle_results'
    utils.create_folder(results_path)

    for task in tasks:
        plots_data = []
        early_exit_accs = []
        early_exit_aucs = []
        orig_accs = []

        for network in networks:
            print('Task {}, Network: {}'.format(task, network))
            _, acc, auc = get_rad_confidence_threshold(path, task, network, rad_limit=15, device='cpu')
            print(f'acc: {acc}, auc: {auc}')


            print('Computing the delayedness for the perfect oracle early exit criteria...')
            plot_data, early_exit_auc, orig_acc, early_exit_acc = get_oracle_latency_plot(path, task, network, device)
            plots_data.append(plot_data)
            early_exit_accs.append(early_exit_acc)
            early_exit_aucs.append(early_exit_auc)
            orig_accs.append(orig_acc)

        plot_save_path = os.path.join(results_path, f'{task}')
        title = f'{task}-'
        for orig_acc, network in zip(orig_accs, networks):
            title = title + f'{network}(ACC {orig_acc:.1f}) '

        draw_plot(plots_data, early_exit_accs, early_exit_aucs, networks, title, plot_save_path)
    '''
