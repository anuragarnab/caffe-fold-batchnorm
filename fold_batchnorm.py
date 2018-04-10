"""Converts Batch Norm and Scale layers into a single Scale layer for inference.
   This uses significantly less memory at inference time for networks that use a lot
   of batch normalisation (such as ResNet).
"""

import argparse
import caffe
import caffe.proto.caffe_pb2 as cp
import google.protobuf as pb
import numpy as np

caffe.set_mode_cpu()

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def_original', type=str, 
        help='Input model definition to convert')
    parser.add_argument('--model_weights_original', type=str, 
        help = 'Input model parameters')
    parser.add_argument('--model_def_folded', type=str, 
        help='Output model definition with batch norm and scale layers' 
             'folded into a single scale layer')
    parser.add_argument('--model_weights_folded', type=str, 
        help='Output model weights with batch norm and scale layers folded together.')
    parser.add_argument('--lr_mult', type=float, default=1.0,
        help='Learning rate multipler of the new, folded scale layer')
    parser.add_argument('--decay_mult', type=float, default=1.0,
        help='Weight decay multipler of the new, folded scale layer')

    args = parser.parse_args()
    return args


def ReadNet(model_def):
    """Reads a .prototxt file that defines the network."""
    with open(model_def) as f:
        net = cp.NetParameter()
        pb.text_format.Parse(f.read(), net)
        return net


def LayersToDict(network):
    """Creates mapping from layer name to its index in the layers array."""
    layers_dict = {}
    for i, layer in enumerate(network.layer):
        if layer.name in layers_dict:
            raise AssertionError('Network should not have duplicate layer names')
        layers_dict[layer.name] = i
        
    return layers_dict


def FindBatchNormLayers(network):
    """Traverse the network and return names of all batch norm layers"""
    batch_norm_keys = []
    for layer in network.layer:
        if layer.type =='BatchNorm':
            batch_norm_keys.append(layer.name)
    
    return batch_norm_keys


def DerefBatchNormLayers(network, batch_norm_names, layers_dict, suffix='_fold', 
                         lr_mult=1.0, decay_mult=1.0):
    """For all batch norm layers, effectively remove them.
       Sets the bottom blob of the layer following batch norm to the top blob of 
       the layer before the batch norm layer.
       This effectively removes the batch norm layer from the graph.
       Also have the option of setting the learning rate multiplier of the new, folded scale layer.
       The multiplier adjustment assumes that there are exactly two parameters in the scale layer
       
       Note that one blob can be referenced by multiple layers, especially 
       in the case of in-place layers. To get around this, we just test to see
       whether the previous and next layers in the "layers" list are actually the 
       previous and next layers of the batch norm layer in question.
       This assumption works for most ResNet definitions, but writing your prototxt
       in a non-linear way would break it.
    """
    for bn_layer_name in batch_norm_names:
        index = layers_dict[bn_layer_name]
        bn_layer = network.layer[index]
        
        if (len(bn_layer.bottom) != 1) or (len(bn_layer.top) != 1):
            raise AssertionError('Expected bn layer to have one top and bottom')
                
        prev_layer_idx = index - 1
        next_layer_idx = index + 1
        prev_layer, next_layer = network.layer[prev_layer_idx], network.layer[next_layer_idx]
        
        if not (prev_layer.top == bn_layer.bottom and bn_layer.top == next_layer.bottom):
            raise AssertionError("Could not find previous and next nodes for"
                                 "batch norm layer")
        
        if next_layer.type != 'Scale':
            print  bn_layer_name, next_layer.type, next_layer.name
            raise AssertionError('Expected Scale layer to follow batch norm layer')
        
        if not (len(prev_layer.top) == 1 and len(next_layer.bottom) == 1):
            raise AssertionError("Expected previous and next blobs to have" 
                                 "only one input and output")
        
        next_layer.bottom[0] = prev_layer.top[0]
        next_layer.name = next_layer.name + suffix

        if lr_mult != 1.0 or decay_mult != 1.0:
            while len(next_layer.param) < 2:
                next_layer.param.add()
            for i in range(len(next_layer.param)):
                next_layer.param[i].lr_mult = lr_mult
                next_layer.param[i].decay_mult = decay_mult


def RemoveBatchNormLayers(network, batch_norm_names):
    """Actually removes batch norm layers from the graph."""
    i = 0
    j = 0
    while i < len(network.layer) and j < len(batch_norm_names):       
        if network.layer[i].name == batch_norm_names[j]:
            del network.layer[i]
            j += 1
        else:
            i += 1
            
    if j != len(batch_norm_names):
        print j, len(batch_norm_names)
        raise AssertionError('All batch norm layers were not removed')


def ConvertWeights(net_orig, net_new, suffix='_fold', eps=1e-5):
    """Merges the batch norm and scale layers.
       This can be done at test time when the batch-norm layer uses a
       fixed mean and variance.
    """
    for layer_name in net_orig.params.keys():
        if layer_name[:2] == 'bn':
            scale_layer_name = layer_name.replace('bn', 'scale')
            
            mu = net_orig.params[layer_name][0].data
            var = net_orig.params[layer_name][1].data
            
            # The standard Caffe implementation uses this, whilst some others do not
            if len(net_orig.params[layer_name]) == 3:
                mov_ave_factor = net_orig.params[layer_name][2].data[0]
                mu = mu * (1 / mov_ave_factor)
                var = var * (1 / mov_ave_factor)
            
            sigma = np.sqrt(var + eps)            
            gamma = net_orig.params[scale_layer_name][0].data
            beta = net_orig.params[scale_layer_name][1].data
            
            gamma_new = gamma / sigma
            beta_new = beta - gamma * mu / sigma

            new_scale_layer_name = scale_layer_name + suffix
            net_new.params[new_scale_layer_name][0].data[...] = gamma_new
            net_new.params[new_scale_layer_name][1].data[...] = beta_new


def main():
    args = ParseArgs()

    network_def = ReadNet(args.model_def_original)
    layers_dict = LayersToDict(network_def)
    batch_norm_names = FindBatchNormLayers(network_def)
    
    DerefBatchNormLayers(network_def, batch_norm_names, layers_dict, 
                         lr_mult=args.lr_mult, decay_mult=args.decay_mult)
    RemoveBatchNormLayers(network_def, batch_norm_names)

    with open(args.model_def_folded, 'w') as f:
        f.write(str(network_def))

    net_orig = caffe.Net(args.model_def_original, args.model_weights_original, caffe.TEST)
    net_folded = caffe.Net(args.model_def_folded, args.model_weights_original, caffe.TEST)

    ConvertWeights(net_orig, net_folded)
    net_folded.save(args.model_weights_folded)


if __name__ == '__main__':
    main()