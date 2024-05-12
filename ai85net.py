###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Networks that fit into AI84

Optionally quantize/clamp activations
"""
from torch import nn

import ai8x


class AI85Net5(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(32, 32),
                 planes=64, pool=2, fc_inputs=8, bias=False, **kwargs):    
        super().__init__()

        # 8 neurons --> fc_inputs
        # 64 filters --> planes

        # Verifications to ensure that the parameters don't exceed the device's limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = 32
        
        # LAYER 0: Convolutional layer with integrated ReLU activation
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3, padding=1, bias=bias, **kwargs)
        
        # LAYER 1: Pooling layer that also includes convolution and ReLU
        self.pool1 = ai8x.FusedMaxPoolConv2dReLU(planes, 64, 3, pool_size=pool, pool_stride=2, padding=0, bias=bias, **kwargs)
        
        dim //= 2  # Update dimension after pooling
        flattened_dim = dim * dim
        print('Flattened dim is: ',flattened_dim)
        
        # LAYER 2:
        self.fc1 = ai8x.Linear(flattened_dim, fc_inputs, bias=True, wide=True, **kwargs)

        # LAYER 3:
        self.fc2 = ai8x.Linear(fc_inputs, num_classes, bias=True, wide=True, **kwargs)

        # Weight initialization for the convolutional layers using Kaiming He Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # Defines the propagation forward
    def forward(self, x):           # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)           # Applies the 1st convolutional layer and ReLU
        x = self.pool1(x)   
        x = x.view(x.size(0), -1)   # Smoothes the tensor for the completly connected layer
        x = self.fc1(x)             # 1st layer completely connected
        x = self.fc2(x)             # 2nd layer completely connected
        return x


def ai85net5(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net5(**kwargs)   # Creates and returns a AI85Net5 instance with the provided arguments


class AI85NetExtraSmall(nn.Module):
    """
    Minimal CNN for minimum energy per inference for MNIST
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 fc_inputs=8, bias=False, **kwargs):
        super().__init__()

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 8, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> 8x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(8, 8, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 8x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 8x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8, fc_inputs, 3,
                                                 pool_size=4, pool_stride=4, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 4  # pooling, padding 0 -> 8x4x4
        # padding 1 -> 8x4x4

        self.fc = ai8x.Linear(fc_inputs*dim*dim, num_classes, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85netextrasmall(pretrained=False, **kwargs):
    """
    Constructs a AI85NetExtraSmall model.
    """
    assert not pretrained
    return AI85NetExtraSmall(**kwargs)


models = [
    {
        'name': 'ai85net5',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85netextrasmall',
        'min_input': 1,
        'dim': 2,
    },
]
