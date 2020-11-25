import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim

import time
import utils
from tqdm import tqdm


def craft_per_sample_perturb_attack( \
    model, test_loader, \
    max_iter=250, per_iter=10, epsilon=8, eps_step=0.5, grad_sparsity=99, nbatch=10, device='cpu'):

    print (' [DeepSloth - L1] start crafting per-sample attacks...')
    model.eval()

    # max iter and save parameters
    attack_data_iters = [list() for _ in range(int((max_iter/per_iter))+2)] # returns unperturbed, noisy and the perturbed for every per_iter iterations
    attack_labels = []

    # to measure the time
    start_time = time.time()

    # loop over the test set
    for batch_idx, batch in enumerate(test_loader):
        # if nbatch > 0 and batch_idx == nbatch: break

        attack_labels.extend(batch[1])

        # : original dataset
        b_x = batch[0].to(device, dtype=torch.float)
        if b_x.min() < 0 or b_x.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        r = np.random.laplace(size=b_x.shape)
        norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1, ord=1).reshape(-1, 1, 1, 1)
        delta = torch.from_numpy((r / norm) * epsilon).to(device).float()

        attack_data_iters[0].append(b_x.cpu().detach().numpy())  # unperturbed
        attack_data_iters[1].append(torch.clamp(b_x + delta, min=0, max=1).cpu().detach().numpy())   # noisy

        adv = torch.clamp((b_x + delta), min=0, max=1).clone().detach().requires_grad_(True)

        for ii in tqdm(range(1, max_iter+1), desc='[DeepSloth-{}]'.format(batch_idx)):
            adv = adv.clone().detach().to(torch.float).requires_grad_(True)

            if ii % per_iter == 0:
                attack_data_iters[int(ii/per_iter) + 1].append(adv.cpu().detach().numpy())  # perturbed

            ce_loss = 0
            logits = model(adv)[:-1]
            for ic_logit in logits:
                layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                ce_loss += layer_cost

            # Define gradient of loss wrt input
            grad, = torch.autograd.grad(-ce_loss, [adv]) # there is a negative sign because we want to decrease the loss (take step opposite direction to the gradient)
            grad_view = grad.view(grad.shape[0], -1)
            abs_grad = torch.abs(grad_view)

            k = int(grad_sparsity/100.0 * abs_grad.shape[1])
            percentile_value, _ = torch.kthvalue(abs_grad, k, keepdim=True)

            percentile_value = percentile_value.repeat(1, grad_view.shape[1])
            tied_for_max = torch.ge(abs_grad, percentile_value).int().float()
            num_ties = torch.sum(tied_for_max, dim=1, keepdim=True)

            optimal_perturbation = (torch.sign(grad_view) * tied_for_max) / num_ties
            optimal_perturbation = optimal_perturbation.view(grad.shape)

            # Add perturbation to original example to obtain adversarial example
            adv = adv + optimal_perturbation * eps_step
            adv = torch.clamp(adv, 0, 1)

            # Clipping perturbation eta to the l1-ball
            delta = adv - b_x
            delta = delta.renorm(p=1, dim=0, maxnorm=epsilon)
            adv = torch.clamp(b_x + delta, 0, 1)

            del ce_loss, logits

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth - L1] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data_iters = [np.vstack(attack_data) for attack_data in attack_data_iters]
    attack_labels = np.asarray(attack_labels)
    return attack_data_iters, attack_labels


def craft_universal_perturb_attack( \
    model, train_loader, \
    max_epoch=1, max_iter=100, step_size=0.5, epsilon=8, grad_sparsity=90, nbatch=10, \
    device='cpu', optimize=False):

    # configurations
    # step_size_schedule = [step_size]* int((max_epoch)/3) + [step_size/2]*int((max_epoch)/3)+ [step_size/10]*int(max_epoch/3)
    step_size_schedule = [step_size] * max_epoch

    print (' [DeepSloth - L1] start crafting universal perturbation...')
    model.eval()

    perturbs = []

    no_perturb = torch.FloatTensor(torch.Size((3, model.input_size, model.input_size))).zero_()
    perturbs.append(no_perturb.cpu().detach().numpy())


    r = np.random.laplace(size=(1, 3, model.input_size, model.input_size))
    norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1, ord=1).reshape(-1, 1, 1, 1)
    delta = torch.from_numpy((r / norm) * epsilon).to(device).float() # init delta

    delta.to(device, dtype=torch.float)
    delta.requires_grad_(True)

    # no perturb (random noise)
    perturbs.append(delta[0].cpu().detach().numpy())

    # to measure the time
    start_time = time.time()

    for epoch in tqdm(range(0, max_epoch), desc='[Univ-DeepSloth]'):
        # loop over the test set
        for batch_idx, batch in enumerate(train_loader):
            if nbatch > 0 and batch_idx == nbatch: break
            # : original dataset
            b_x = batch[0].to(device, dtype=torch.float)

            for ii in range(max_iter):
                ce_loss = 0
                logits = model(torch.clamp((b_x + delta), min=0, max=1))[:-1]
                for ic_idx, ic_logit in enumerate(logits):
                    if optimize and (ic_idx > 0): break     # optimize for the first exit
                    layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                    ce_loss += layer_cost

                # Define gradient of loss wrt input
                grad, = torch.autograd.grad(-ce_loss, [delta]) # there is a negative sign because we want to decrease the loss (take step opposite direction to the gradient)
                grad_view = grad.view(grad.shape[0], -1)
                abs_grad = torch.abs(grad_view)

                k = int(grad_sparsity/100.0 * abs_grad.shape[1])
                percentile_value, _ = torch.kthvalue(abs_grad, k, keepdim=True)

                percentile_value = percentile_value.repeat(1, grad_view.shape[1])
                tied_for_max = torch.ge(abs_grad, percentile_value).int().float()
                num_ties = torch.sum(tied_for_max, dim=1, keepdim=True)

                optimal_perturbation = (torch.sign(grad_view) * tied_for_max) / num_ties
                optimal_perturbation = optimal_perturbation.view(grad.shape)

                # Add perturbation to original example to obtain adversarial example

                delta = delta + optimal_perturbation * step_size_schedule[epoch]
                delta = delta.renorm(p=1, dim=0, maxnorm=epsilon)
                del ce_loss, logits

        perturbs.append(delta[0].cpu().detach().numpy())

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth L1 Universal] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # end for epoch...
    return perturbs
