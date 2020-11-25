"""
    Our attack - DeepSloth (per-sample / universal)
    [Note: the patch attack doesn't work]
"""
# basic
import time
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.autograd import Variable

# custom libs
import utils


# craft on the first batch
def craft_universal_perturb_attack( \
    model, train_loader, \
    max_epoch=1, max_iter=20, step_size=0.005, epsilon=0.03125, nbatch=1, \
    device='cpu', optimize=False):
    print(' [DeepSloth - Linf] start crafting universal attacks...')
    # Note: max_epoch must be divisible for 3 for the step size schedule

    model.eval()
    perturbs = []

    no_perturb = torch.FloatTensor(torch.Size((3, model.input_size, model.input_size))).zero_()
    perturbs.append(no_perturb.cpu().detach().numpy())

    # initial perturbation
    perturb = torch.FloatTensor(torch.Size((3, model.input_size, model.input_size))).to(device).uniform_(-epsilon, epsilon)
    perturbs.append(perturb.cpu().detach().numpy())

    # configurations
    # step_size_schedule = [step_size]*int((max_epoch)/3) + [step_size/10]*int((max_epoch)/3)+ [step_size/100]*int(max_epoch/3)
    step_size_schedule = [step_size]

    # to measure the time
    start_time = time.time()

    # loop over the epochs
    for epoch in tqdm(range(0, max_epoch), desc='[Univ. DeepSloth]'):
        # print(f'epoch {epoch} - step size {step_size_schedule[epoch]}')
        for batch_idx, batch in enumerate(train_loader):
            # print(f'batch {batch_idx}')
            if nbatch > 0 and batch_idx == nbatch: break

            # : original dataset
            b_x = batch[0].to(device, dtype=torch.float)
            ori_b_x = b_x.data

            # : do perturbations
            for _ in range(max_iter):
                perturb.requires_grad = True
                cost = 0
                b_x = ori_b_x + perturb.repeat(b_x.shape[0],1,1,1)
                logits = model(b_x)[:-1]
                model.zero_grad()
                for ic_idx, ic_logit in enumerate(logits):
                    if optimize and (ic_idx > 0): break     # optimize for the first exit
                    layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                    cost += layer_cost
                cost.backward(retain_graph=True)

                # : compute based on the norm
                perturb = perturb - step_size_schedule[epoch]*perturb.grad.sign()
                perturb = torch.clamp(perturb, min=-epsilon, max=epsilon).detach_()


                del cost, logits
            # : end for _

        perturb_numpy = perturb.cpu().detach().numpy()
        perturbs.append(perturb_numpy)

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth L-inf] time taken for crafting one perturbation: {:.4f}'.format(termi_time - start_time))

    # end for epoch...
    return perturbs

def apply_perturb_attack(test_loader, perturb, images_save_path=None, device='cpu'):

    attack_data = []
    attack_labels = []

    perturb_tensor = torch.from_numpy(perturb).to(device, dtype=torch.float)
    saved = False
    for batch in test_loader:
        b_x = batch[0].to(device, dtype=torch.float)
        perturbed = b_x + perturb_tensor.repeat(b_x.shape[0],1,1,1)
        perturbed = torch.clamp(perturbed, min=0, max=1)
        attack_data.append(perturbed.cpu().detach().numpy())
        attack_labels.extend(batch[1])

        if saved is False and images_save_path is not None:
            vis_tensor = torch.stack((b_x,perturbed), dim=1).view(perturbed.shape[0]+b_x.shape[0], *b_x.shape[1:])
            utils.save_batch_of_tensor_images(images_save_path, vis_tensor)

            saved = True

    attack_data, attack_labels =  np.vstack(attack_data), np.asarray(attack_labels)

    return attack_data, attack_labels

def craft_per_sample_perturb_attack( \
    model, test_loader, \
    max_iter=30, per_iter=10, epsilon=0.03125, eps_step=0.002, nbatch=10, device='cpu'):
    print (' [DeepSloth - Linf] start crafting per-sample attacks...')
    model.eval()

    # max iter and save parameters
    attack_data_iters = [list() for _ in range(int((max_iter/per_iter))+2)]
    attack_labels = []

    # to measure the time
    start_time = time.time()

    # loop over the test set
    for batch_idx, batch in enumerate(test_loader):
        # if nbatch > 0 and batch_idx == nbatch: break

        # : original dataset
        b_x = batch[0].to(device, dtype=torch.float)
        attack_data_iters[0].append(b_x.cpu().detach().numpy())  # unperturbed

        ori_b_x = b_x.data
        b_x = b_x + torch.FloatTensor(b_x.shape).to(device).uniform_(-eps_step, eps_step)
        attack_data_iters[1].append(b_x.cpu().detach().numpy())  # with l_inf noise

        # : do perturbations
        for ii in tqdm(range(1, max_iter+1), desc='[DeepSloth-{}]'.format(batch_idx)):
            b_x.requires_grad = True
            cost = 0
            logits = model(b_x)[:-1]
            model.zero_grad()
            for ic_logit in logits:
                layer_cost = utils.CrossEntropyLogitsWithUniform(ic_logit)
                cost += layer_cost
            cost.backward(retain_graph=True)

            adv_cur_x = b_x - eps_step*b_x.grad.sign()
            perturb = torch.clamp(adv_cur_x - ori_b_x, min=-epsilon, max=epsilon)
            b_x = torch.clamp(ori_b_x + perturb, min=0, max=1).detach_()

            del cost, logits

            if ii % per_iter == 0:
                attack_data_iters[int(ii/per_iter)+1].append(b_x.cpu().detach().numpy())

        attack_labels.extend(batch[1])

    # to measure the time
    termi_time = time.time()
    print (' [DeepSloth - Linf] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data_iters = [np.vstack(attack_data) for attack_data in attack_data_iters]
    attack_labels = np.asarray(attack_labels)
    return attack_data_iters, attack_labels

def craft_per_sample_perturb_attack_on_samples( \
    model, samples, \
    max_iter=30, epsilon=0.03125, eps_step=0.002, device='cpu'):
    # set the model to eval
    model.eval()

    # convert to cuda...
    x_adv = samples.clone()
    if 'cuda' == device: x_adv = x_adv.cuda()

    # create an adv. example w. random init.
    x_org = x_adv.data
    x_rnd = torch.FloatTensor(x_adv.shape).uniform_(-eps_step, eps_step)
    if 'cuda' == device: x_rnd = x_rnd.cuda()
    x_adv += x_rnd
    x_adv = Variable(x_adv, requires_grad=True)

    # do perturbations
    for ii in range(1, max_iter+1):
        loss = 0
        logits = model(x_adv)[:-1]
        for ic_logit in logits:
            loss += utils.CrossEntropyLogitsWithUniform(ic_logit)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv.data.add_(-eps_step * torch.sign(grad.data))
        x_per = torch.clamp(x_adv - x_org, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_org + x_per, min=0, max=1)

    # return the perturbed samples
    return x_adv
