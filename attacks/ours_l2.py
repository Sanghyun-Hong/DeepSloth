import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim


import time
import utils
from tqdm import tqdm


def craft_per_sample_perturb_attack( \
    model, test_loader, \
    max_iter=550, per_iter=50, gamma=0.1, init_norm=1., nbatch=10, device='cpu'):
    print (' [DeepSloth - L2] start crafting per-sample attacks...')
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

        attack_data_iters[0].append(b_x.cpu().detach().numpy())  # unperturbed

        r = np.random.randn(*b_x.shape)
        norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
        delta =  torch.from_numpy((r / norm) * init_norm).to(device).float() # init delta
        delta.requires_grad_(True)


        noise = torch.from_numpy(  (r / norm) * (init_norm * ((1 - gamma) ** (int(max_iter/per_iter) - 1)))).to(device).float()
        attack_data_iters[1].append(torch.clamp(b_x +  noise, min=0, max=1).cpu().detach().numpy())   # random noise

        cur_norm = init_norm

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0.01)

        for ii in tqdm(range(1, max_iter+1), desc='[DeepSloth-{}]'.format(batch_idx)):
            delta.requires_grad_(True)

            ce_loss = 0
            logits = model(torch.clamp((b_x + delta), min=0, max=1))[:-1]
            for ic_logit in logits:
                layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                ce_loss += layer_cost

            optimizer.zero_grad()
            ce_loss.backward(retain_graph=True)

            # renorming gradient
            delta.grad = delta.grad.renorm(p=2, dim=0, maxnorm=1)
            optimizer.step()

            # renorm the perturbation
            for idx in range(len(b_x)):
                delta[idx] = delta[[idx]].renorm(p=2, dim=0, maxnorm=cur_norm)[0]

            # divide the perturb norm bound by gamma
            if ii % per_iter == 0:
                adv = torch.clamp((b_x + delta), min=0, max=1)
                attack_data_iters[int(ii/per_iter) + 1].append(adv.cpu().detach().numpy())
                cur_norm = cur_norm * (1 - gamma)

            delta.detach_()
            scheduler.step()

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth - L2] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data_iters = [np.vstack(attack_data) for attack_data in attack_data_iters]
    attack_labels = np.asarray(attack_labels)
    return attack_data_iters, attack_labels


def craft_universal_perturb_attack( \
    model, train_loader, \
    max_epoch=11, max_iter=50, gamma=0.1, init_norm=1., nbatch=5, \
    device='cpu', optimize=False):

    print (' [DeepSloth - L2] start crafting universal perturbation...')
    model.eval()

    perturbs = []

    no_perturb = torch.FloatTensor(torch.Size((3, model.input_size, model.input_size))).zero_()
    perturbs.append(no_perturb.cpu().detach().numpy()) # no perturb

    r = np.random.randn(1, 3, model.input_size, model.input_size)
    norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
    delta =  torch.from_numpy((r / norm) * init_norm).to(device).float() # init delta

    delta.to(device, dtype=torch.float)
    delta.requires_grad_(True)

    noise = (r / norm) * (init_norm * ((1 - gamma) ** (max_epoch - 1))) # this is the perturb norm at the end iteration
    perturbs.append(noise[0]) # (random noise)

    optimizer = optim.SGD([delta], lr=0.1)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: (1 - gamma))

    # to measure the time
    start_time = time.time()

    cur_norm = init_norm

    for _ in tqdm(range(0, max_epoch), desc='[Univ-DeepSloth]'):
        # loop over the test set
        for batch_idx, batch in enumerate(train_loader):
            if nbatch > 0 and batch_idx == nbatch: break

            # : original dataset
            b_x = batch[0].to(device, dtype=torch.float)

            for _ in range(max_iter):
                delta.requires_grad_(True)
                ce_loss = 0
                logits = model(torch.clamp((b_x + delta), min=0, max=1))[:-1]
                for ic_idx, ic_logit in enumerate(logits):
                    if optimize and (ic_idx > 0): break     # optimize for the first exit
                    layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                    ce_loss += layer_cost

                optimizer.zero_grad()
                ce_loss.backward(retain_graph=True)

                # renorming gradient
                delta.grad = delta.grad.renorm(p=2, dim=0, maxnorm=1)

                optimizer.step()

                # renorm the perturbation
                delta[0] = delta[[0]].renorm(p=2, dim=0, maxnorm=cur_norm)[0]
                delta.detach_()



        # divide the perturb norm bound by gamma
        cur_norm = cur_norm * (1 - gamma)
        perturbs.append(delta[0].cpu().detach().numpy())

        scheduler.step()

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth L2 Universal] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # end for epoch...
    return perturbs
