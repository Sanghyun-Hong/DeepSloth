import torch
import torch.nn as nn
import numpy as np
import math
import copy

# custom libs
import utils
import model_funcs as mf


class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1


        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1, stride=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))

        self.layers = nn.Sequential(*conv_layers)


        if add_output:
            self.output = utils.InternalClassifier(input_size, output_channels, num_classes)
            self.no_output = False

        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True


    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        fwd = self.layers(x)
        return fwd, 0, None

class FcBlockWOutput(nn.Module):
    def __init__(self, fc_params, output_params, flatten=False):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]

        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(utils.Flatten())

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None


class VGG_SDN(nn.Module):
    def __init__(self, params):
        super(VGG_SDN, self).__init__()

        self.output_to_return_when_ICs_are_delayed = 'network_output' # 'network_output' or 'most_confident_output'

        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels'] # the first element is input dimension
        self.fc_layer_sizes = params['fc_layers']

        # read or assign defaults to the rest
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.add_output = params['add_output']

        # for vanilla training
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        # for AT
        self.advtrain_func = mf.sdn_advtrain
        self.advtrain_func = mf.sdn_advtrain

        self.num_output = sum(self.add_output) + 1

        self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        self.init_depth = 0
        self.end_depth = 2

        # add conv layers
        input_channel = 3
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            output_id += add_output

        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            fc_input_size = width
            output_id += add_output

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


    # anytime prediction, only one image
    def anytime(self, x, label):
        # start
        ic_num = 0
        foward = self.init_conv(x)

        # loop over the entire layers
        for layer in self.layers:
            foward, has_ic, output = layer(foward)
            if has_ic:
                predict = output.argmax(dim=1, keepdim=True)[0]

                # : return when made a correct prediction
                if (predict.item() == label.item()):
                    return output, ic_num
                ic_num += has_ic

        output = self.end_layers(foward)
        return output, ic_num


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
