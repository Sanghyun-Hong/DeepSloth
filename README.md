## [ICLR 2021: Spotlight] DeepSloth

This repository contains the code for reproducing the results in our paper:

- [A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference](https://arxiv.org/abs/2010.02432) **[ICLR 2021: Spotlight]**
- **[Sanghyun Hong](https://secure-ai.systems)**, **Yiğitcan Kaya**, Ionuţ-Vlad Modoranu, Tudor Dumitraş. (Bold: Equal contributions)

---

### TL; DR

Is the computational savings provided by the input-adaptive 'multi-exit architectures' robust against adversarial perturbations? No.


### Abstract (Tell me more!)

Recent increases in the computational demands of deep neural networks (DNNs), combined with the observation that most input samples require only simple models, have sparked interest in _input-adaptive_ multi-exit architectures, such as MSDNets or Shallow-Deep Networks. These architectures enable faster inferences and could bring DNNs to low-power devices, e.g. in the Internet of Things (IoT). However, it is unknown if the computational savings provided by this approach are robust against adversarial pressure. In particular, an adversary may aim to slow down adaptive DNNs by increasing their average inference time—a threat analogous to the _denial-of-service_ attacks from the Internet. In this paper, we conduct a systematic evaluation of this threat by experimenting with three generic multi-exit DNNs (based on VGG16, MobileNet, and ResNet56) and a custom multi-exit architecture, on two popular image classification benchmarks (CIFAR-10 and Tiny ImageNet). To this end, we show that adversarial sample-crafting techniques can be modified to cause slowdown, and we propose a metric for comparing their impact on different architectures. We show that a slowdown attack reduces the efficacy of multi-exit DNNs by 90%–100%, and it amplifies the latency by 1.5–5× in a typical IoT deployment. We also show that it is possible to craft universal, reusable perturbations and that the attack can be effective in realistic black-box scenarios, where the attacker has limited knowledge about the victim. Finally, we show that adversarial training provides limited protection against slowdowns. These results suggest that further research is needed for defending multi-exit architectures against this emerging threat.

---

### Contents

1. [Pre-requisites](#pre-requisites)
2. [Training Models](#training-models)
3. [Crafting Adversarial Examples](#crafting-adversarial-examples)
4. [Visualize Internal Representations](#visualize-internal-representations)
5. [Run Transferability Experiments](#run-transferability-experiments)
6. [Run Adversarial Training](#run-adversarial-training)

&nbsp;

---

### Pre-requisites

Download the TinyImageNet dataset from this [link](https://tiny-imagenet.herokuapp.com/), and unzip the downloaded file under `datasets/originals`. The following command will help.

```
  $ mkdir -p datasets/originals
  $ unzip tiny-imagenet-200.zip datasets/originals/
  $ python datasets.py
```

----

### Training Models

You can use the following script to train multi-exit models (SDNs).

```
  $ python train_sdns.py \
    --dataset <cifar10 or tinyimagenet> \
    --network <vgg16bn, resnet56, or mobilenet> \
    --vanilla <set if you want the training of vanilla models> \
    --ic-only <set if you want to train the internal classifiers with the network> \
    --adv-run <set if you want the adversarial training> \
    --attacks <with adv-run, PGD, PGD-avg, or PGD-max> \
    --maxiter <the number of iterations for adversarial training> \
    --epsilon <the perturbation limit in l_inf> \
    --epsstep <the perturbation step>
```

The trained model will be stored under the `models` folder.
(e.g. `models/<dataset>/<dataset>_<network>_<nettype>`)


----

### Crafting Adversarial Examples

To craft DeepSloth adversarial samples, you can use the following script. (Note: to run the attacks on the adversarially-trained models, you need to use `run_attacks_atnet.py`.)

```
  $ python run_attacks.py \
    --dataset <cifar10 or tinyimagenet> \
    --network <vgg16bn, resnet56, or mobilenet> \
    --nettype <cnn, sdn_ic_only, or PGD_10_8_2_cnn> \
    --runmode <attack or analysis> \
    --attacks <PGD, PGD-avg, PGD-max, UAP, ours, or ours-class> \
    --ellnorm <linf, l1 or l2> \
    --nsample <# samples - for UAP> \
    --batch-size <default: 250>
```

You first need to run the attack for each model with `--runmode attack`. This will craft the adversarial samples and save the adversarial examples to a file. Then, with `--runmode analysis`, you can load a model and an attack file for testing the effectiveness of the attack.

&nbsp;

Below, we show a few sample commands to craft different attacks:

#### Vanilla Adversarial Examples

To run conventional attacks on SDNs with various norms, you can refer to the examples below.

```
  // attacks (PGD, PGD-avg, PGD-max, UAP)
  $ python run_attacks.py \
   --dataset cifar10 \
   --network vgg16bn \
   --nettype sdn_ic_only \
   --attacks PGD \ # change the attack type here
   --ellnorm linf \
   --runmode attack # you can use analysis for the second run
```


#### Per-Sample / Universal DeepSloth Adversarial Examples

You can run DeepSloth on SDNs. Please refer to the following example. `ours` is for PerSample/Universal DeepSloth, and `ours-class` is for Class-Universal DeepSloth.

```
  // DeepSloth (ours)
  $ python run_attacks.py \
    --dataset cifar10 \
    --batch-size 250 \
    --network vgg16bn \
    --nettype sdn_ic_only \
    --attacks ours \ # ours or ours-class
    --ellnorm linf \
    --runmode attack
```


#### Per-Class DeepSloth Adversarial Examples

You can run per-class DeepSloth on SDNs with various norms, as follows:

```
  // DeepSloth (ours, w. different norms)
  $ python run_attacks.py \
    --dataset cifar10 \
    --batch-size 250 \
    --network vgg16bn \
    --nettype sdn_ic_only \
    --attacks ours-class \ # ours-class for Class-Universal DeepSloth
    --ellnorm linf \
    --runmode attack
```


----

### Visualize Internal Representations

To run the analysis of internal representation in Sec 5.1, you can use the following script.

```
  $ python run_analysis.py \
    --dataset <cifar10 or tinyimagenet> \
    --network <vgg16bn, resnet56, or mobilenet> \
    --nettype <cnn, sdn, or sdn_ic_only: the network types> \
    --o-class <the class of our interest: dog - for example> \
    --nsample <the # of samples consider in the class> \
```

The analysis results will be stored under the analysis folder.

```
  - store location: analysis/<dataset>/<netname>/<nsamples>
  - the stored files are:
    - Adversarial samples as PNG files.
    - Internal representation per layer as a PNG file.
```


----

### Run Transferability Experiments

To run the transferability experiments in Sec 5.2, you can use the following script.

```
  python run_transferability_experiments.py
```

Unfortunately, the arguments required to run this script is hardcoded in the Python script.

```
  [From the line 40 to 45 in the script]
   - task: the same as the dataset (cifar10 or tinyimagenet)
   - proxy_network_arc: the surrogate model
   - victim_network_arc: the victim model
   - rad_limit: the stopping criteria of the victim network
   - scenario:
     - scenario_1: when attacker knows the fraction of the training dataset
     - scenario_2: when attacker uses a different datset
```

The experimental results will be stored with the EEC plots as follows.

```
  - store location: <scenario>/<dataset>/<proxy-netname>...
  - the stored files are:
    - A pickle file contains a dictionary with the analysis data
    - A plot shows the plot like EEC.
```


----

### Run Adversarial Training

To run the adversarial training in Sec 5, you first need to train the AT-CNN model by using the following script.

```
  $ python train_ours.py \
    --dataset cifar10 \
    --network vgg16bn \
    --cnn --cnn-adv --sdn --sdn-adv \
    --attacks PGD --maxiter 10 --epsilon 8 --epsstep 2
```

&nbsp;

To train AT-models on PGD-avg. or PGD-max. on top of the AT-CNN model, you can use the following scripts.

```
  // copy the AT-CNN to the correct locations (prepare)
  $ cp -r models/cifar10/cifar10_vgg16bn_adv_adv_PGD_10_8_2_cnn \
        models/cifar10/cifar10_vgg16bn_adv_adv_PGD-avg_10_8_2_cnn

  // train with PGD-avg.
  $ python train_ours.py \
    --dataset cifar10 \
    --network vgg16bn \
    --cnn-adv --sdn --sdn-adv \
    --attacks PGD-avg --maxiter 10 --epsilon 8 --epsstep 2

  // copy the AT-CNN to the correct locations (prepare)
  $ cp -r models/cifar10/cifar10_vgg16bn_adv_adv_PGD_10_8_2_cnn \
        models/cifar10/cifar10_vgg16bn_adv_adv_PGD-max_10_8_2_cnn

  // train with PGD-max
  $ python train_ours.py \
    --dataset cifar10 \
    --network vgg16bn \
    --cnn-adv --sdn --sdn-adv \
    --attacks PGD-max --maxiter 10 --epsilon 8 --epsstep 2
```

&nbsp;

To train AT-models on DeepSloth, on top of the AT-CNN model, you can use the similar scripts as follows:

```
  // copy the AT-CNN to the correct locations (prepare)
  $ cp -r models/cifar10/cifar10_vgg16bn_adv_adv_PGD_10_8_2_cnn \
        models/cifar10/cifar10_vgg16bn_adv_adv_ours_10_8_2_cnn

  // train with DeepSloth
  python train_ours.py \
    --dataset cifar10 \
    --network vgg16bn \
    --cnn-adv --sdn --sdn-adv \
    --attacks ours --maxiter 10 --epsilon 8 --epsstep 2

  // copy the AT-CNN to the correct locations (prepare)
  $ cp -r models/cifar10/cifar10_vgg16bn_adv_adv_PGD_10_8_2_cnn \
        models/cifar10/cifar10_vgg16bn_adv_adv_mixs_10_8_2_cnn

  // train with DeepSloth + PGD
  $ python train_ours.py \
    --dataset cifar10 \
    --network vgg16bn \
    --cnn-adv --sdn --sdn-adv \
    --attacks mixs --maxiter 10 --epsilon 8 --epsstep 2
```

---

### Cite Our Work

Please cite our work if you find our work is helpful.

```
@inproceedings{Hong2021DeepSloth,
    title={A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference},
    author={Sanghyun Hong and Yigitcan Kaya and Ionuț-Vlad Modoranu and Tudor Dumitras},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=9xC2tWEwBD}
}
```

---

&nbsp;

Please contact [Sanghyun Hong](mailto:sanghyun.hong@oregonstate.edu) for any questions and recommendations.

