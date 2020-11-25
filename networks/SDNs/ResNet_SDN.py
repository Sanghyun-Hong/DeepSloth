import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# custom libs
import utils
import model_funcs as mf


class BasicBlockWOutput(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, params, stride=1):
        super(BasicBlockWOutput, self).__init__()
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU())

        self.layers = layers

        if add_output:
            self.output = utils.InternalClassifier(input_size, self.expansion*channels, num_classes)
            self.no_output = False

        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd = fwd + self.layers[1](x) # shortcut
        return self.layers[2](fwd), 1, self.output(fwd)         # output layers for this module

    def only_output(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd = fwd + self.layers[1](x) # shortcut
        fwd = self.layers[2](fwd) # activation
        out = self.output(fwd)         # output layers for this module
        return out

    def only_forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd = fwd + self.layers[1](x) # shortcut
        return self.layers[2](fwd), 0, None # activation


class ResNet_SDN(nn.Module):
    def __init__(self, params):
        super(ResNet_SDN, self).__init__()

        self.output_to_return_when_ICs_are_delayed = 'network_output' # 'network_output' or 'most_confident_output'

        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']
        self.add_out_nonflat = params['add_output']
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]
        self.init_weights = params['init_weights']
        # for vanilla training
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        # for AT
        self.advtrain_func = mf.sdn_advtrain
        self.advtrain_func = mf.sdn_advtrain
        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        if self.block_type == 'basic':
            self.block = BasicBlockWOutput

        init_conv = []

        if self.input_size ==  32: # cifar10
            self.cur_input_size = self.input_size
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else: # tiny imagenet
            self.cur_input_size = int(self.input_size/2)
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))

        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))

        self.cur_input_size = int(self.cur_input_size/2)
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(utils.Flatten())
        end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]
            params  = (add_output, self.num_classes, int(self.cur_input_size), self.cur_output_id)
            layers.append(self.block(self.in_channels, channels, params, stride))
            self.in_channels = channels * self.block.expansion
            self.cur_output_id += add_output

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, internal=False):
        # forward pass w/o internal representations
        if not internal:
            outputs = []
            fwd = self.init_conv(x)
            for layer in self.layers:
                fwd, is_output, output = layer(fwd)
                if is_output:
                    outputs.append(output)
            fwd = self.end_layers(fwd)
            outputs.append(fwd)
            return outputs
        # forward pass w. internal representations
        else:
            outints = []
            outputs = []
            fwd = self.init_conv(x)
            for layer in self.layers:
                fwd, is_output, output = layer(fwd)
                if is_output:
                    outints.append(fwd.detach().numpy())    # remove .numpy() if you don't want
                    outputs.append(output)
            fwd = self.end_layers(fwd)
            outputs.append(fwd)
            return outputs, outints

    def forward_w_acts(self, x):
        acts = []

        out = self.init_conv(x)

        for layer in self.layers:
            out = layer.only_forward(out)[0]
            acts.append(out)

        out = self.end_layers(out)

        return acts, out


    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []

        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)

                confidence = torch.max(softmax)
                confidences.append(confidence)

                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early

                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        # 'network_output' or 'most_confident_output'
        if self.output_to_return_when_ICs_are_delayed == 'most_confident_output':
            softmax = nn.functional.softmax(output[0], dim=0)
            confidence = torch.max(softmax)
            confidences.append(confidence)
            max_confidence_output = np.argmax(confidences)
            is_early = False
            return outputs[max_confidence_output], max_confidence_output, is_early
        elif self.output_to_return_when_ICs_are_delayed == 'network_output':
            is_early = False
            return output, output_id, is_early
        else:
            raise RuntimeError('Invalid value for "output_to_return_when_not_confident_enough": it should be "network_output" or "most_confident_output"')

    def get_prediction_and_confidences(self, x):
        outputs = self.forward(x.cpu())

        logits, confs, max_confs, labels = [], [], [], []
        for out in outputs:
            softmax = nn.functional.softmax(out[0].cpu(), dim=0)
            pred_label = out.max(1)[1].item()
            pred_conf = torch.max(softmax).item()

            logits.append(list(out[0].cpu().detach().numpy()))
            confs.append(list(softmax.cpu().detach().numpy()))
            max_confs.append(pred_conf)
            labels.append(pred_label)
        return logits, confs, max_confs, labels
