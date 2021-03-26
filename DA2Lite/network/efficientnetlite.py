from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'out_expand_ratio': 6,
    'strides': 2
}, {
    'kernel_size': 5,
    'repeats': 1,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'out_expand_ratio': 6,
    'strides': 2
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'out_expand_ratio': 6,
    'strides': 2
},{
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'out_expand_ratio': 6,
    'strides': 1
},{
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'out_expand_ratio': 6,
    'strides': 2
} ]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

class EfficientNetLite():
    def __init__()
def EfficientNet(
                width_coefficient,
                depth_coefficient,
                depth_divisor=8,
                activation='relu6',
                blocks_args='default',
                model_name='efficientnet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                out_filters=None,
                classes=1000,
                classifier_activation='softmax',
                ):

    def round_filters(filters, divisor = depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))
    

    def __inis

    out = layers.Conv2D(
                        32,
                        3,
                        strides=2,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='stem_conv1')(inputs)
    out = layers.BatchNormalization(axis=bn_axis, name='stem_bn1')(out)
    out = layers.ReLU(max_value=6 , name='stem_ac1')(out)
    
    out = layers.DepthwiseConv2D(
                                    3,
                                    strides=1,
                                    padding='same',
                                    use_bias=False,
                                    depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                    name='stem_dwconv2')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='stem_bn2')(out)
    out = layers.ReLU(max_value=6 , name='stem_ac2')(out)
    
    out = layers.Conv2D(
                        round_filters(16),
                        1,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='stem_conv3')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='stem_bn3')(out)
    out = layers.ReLU(max_value=6 , name='stem_ac3')(out)
    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS
    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0

    block_list = []

    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            out =  block( out,
                            activation,
                            name='block{}{}_'.format(i + 1, chr(j + 97)),
                            **args)
            b += 1

    out = layers.Conv2D(
                        blocks_args[-1]['filters_out'] *6 ,
                        1,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='head_conv1')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='head_bn1')(out)
    out = layers.ReLU(max_value=6 , name='head_activation1')(out)



    out = layers.DepthwiseConv2D(
                                3,
                                strides=1,
                                padding='same',
                                use_bias=False,
                                depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                name='head_dwconv2')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='head_bn2')(out)
    out = layers.ReLU(max_value=6 , name='head_activation2')(out)

    out = layers.Conv2D(
                        round_filters(320),
                        1,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='head_conv3')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='head_bn3')(out)

    out = layers.Conv2D(
                        1280,
                        1,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='head_conv4')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='head_bn4')(out)
    out = layers.ReLU(max_value=6 , name='head_activation3')(out)

    out = layers.AveragePooling2D(pool_size=(out.shape[1],out.shape[2]), name='avg_pool')(out)
    out = layers.Flatten(name='flatten')(out)
    out = layers.Dense(units=classes,kernel_initializer='he_normal')(out)
    out = layers.Activation('softmax', name='softmax')(out)

    # Create model.
    model = training.Model(inputs, out, name=model_name)

    return model

def block(  inputs,
            activation='relu6',
            name='',
            filters_in=32,
            filters_out=16,
            kernel_size=3,
            strides=1,
            expand_ratio=1,
            out_expand_ratio=6):
        
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    filters = filters_in * expand_ratio
    filters_mid = filters_out * out_expand_ratio 
    
    if filters_in != filters_out:
        out = layers.Conv2D(
                            filters,
                            1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'expand_conv')(inputs)
        out = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(out)
        out = layers.ReLU(max_value=6 , name=name+'expand_activation1')(out)


        out = layers.DepthwiseConv2D(
                                    kernel_size,
                                    strides=strides,
                                    padding='same',
                                    use_bias=False,
                                    depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                    name=name + 'dwconv')(out)

        out = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn2')(out)
        out = layers.ReLU(max_value=6 , name=name+'expand_activation2')(out)

        out = layers.Conv2D(filters_out,
                                        1,
                                        padding='same',
                                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                                        name=name + 'expand_conv2')(out)
        out = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn3')(out)
    else:
        out = inputs

    tmp_out = out
    out = layers.Conv2D(filters_mid,
                                    1,
                                    padding='same',
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                                    name=name + 'se_reduce')(out)
    out = layers.BatchNormalization(axis=bn_axis, name=name + 'bn1')(out)
    out = layers.ReLU(max_value=6 , name=name+'ac1')(out)


    out = layers.DepthwiseConv2D(
                            kernel_size,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            depthwise_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'dwconv2')(out)
    out = layers.BatchNormalization(axis=bn_axis, name=name + 'bn2')(out)
    out = layers.ReLU(max_value=6 , name=name+'ac2')(out)

    out = layers.Conv2D(filters_out,
                        1,
                        padding='same',
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'se_expand')(out)

    out = layers.BatchNormalization(axis=bn_axis, name=name + 'bn3')(out)
    out = layers.ReLU(max_value=6 , name=name+'ac3')(out)
    
    out = layers.Add(name=name+ 'add')([out, tmp_out])

    return out

            


def EfficientNetB0_M(input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    return EfficientNet(
                        1.0,
                        1.0,
                        model_name='efficientnetb0_mobile',
                        input_shape=[224,224,3],
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        classifier_activation=classifier_activation,
                        **kwargs)


def EfficientNetB1_M(input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    return EfficientNet(
                        1.0,
                        1.1,
                        model_name='efficientnetb1_mobile',
                        input_shape=[240,240,3],
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        classifier_activation=classifier_activation,
                        **kwargs)

def EfficientNetB2_M(input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    return EfficientNet(
                        1.1,
                        1.2,
                        model_name='efficientnetb2_mobile',
                        input_shape=[260,260,3],
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        classifier_activation=classifier_activation,
                        **kwargs)


def EfficientNetB3_M(input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    return EfficientNet(
                        1.2,
                        1.5,
                        model_name='efficientnetb3_mobile',
                        input_shape=[280,280,3],
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        classifier_activation=classifier_activation,
                        **kwargs)
