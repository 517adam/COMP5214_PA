#  Assignment 3
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # Ensure stride and padding are passed correctly
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    if init_zero_weights:
        layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    """Vanilla‐GAN generator: z → 32×32 RGB (Enhanced Capacity)"""
    def __init__(self, noise_size=100, conv_dim=32): # Increased default conv_dim
        super(DCGenerator, self).__init__()
        # Increase channel depth in intermediate layers
        # z: (B, noise_size, 1, 1) -> (B, conv_dim*8, 4, 4)
        self.deconv1 = deconv(noise_size, conv_dim*4, 4, stride=1, padding=0, batch_norm=True)
        # (B, conv_dim*8, 4, 4) -> (B, conv_dim*4, 8, 8)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 5, stride=1, padding=0, batch_norm=True)
        # (B, conv_dim*4, 8, 8) -> (B, conv_dim*2, 16, 16)
        self.deconv3 = deconv(conv_dim*2, conv_dim*1, 4, stride=2, padding=1, batch_norm=True)
        # (B, conv_dim*2, 16, 16) -> (B, 3, 32, 32)
        self.deconv4 = deconv(conv_dim*1, 3, 4, stride=2, padding=1, batch_norm=False) # Last layer usually no BatchNorm

    def forward(self, z):
        # Apply activation functions AFTER the layer (including BatchNorm)
        out = F.leaky_relu(self.deconv1(z), 0.2)
        out = F.leaky_relu(self.deconv2(out), 0.2)
        out = F.leaky_relu(self.deconv3(out), 0.2)
        # Final layer uses tanh activation
        return F.tanh(self.deconv4(out))


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, init_zero_weights=False):
        super(ResnetBlock, self).__init__()
        # propagate init_zero_weights into the 3×3 conv
        self.conv_layer = conv(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1,
            batch_norm=True,
            init_zero_weights=init_zero_weights
        )

    def forward(self, x):
        return x + self.conv_layer(x)


class CycleGenerator(nn.Module):
    """CycleGAN generator: 32×32 RGB → 32×32 RGB"""
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        # encoder: 32×32×3 → 16×16×conv_dim → 8×8×(conv_dim*2)
        self.conv1 = conv(
            in_channels=3, out_channels=conv_dim, kernel_size=4,
            stride=2, padding=1, batch_norm=True,
            init_zero_weights=init_zero_weights
        )  # B×cd×16×16
        self.conv2 = conv(
            in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=4,
            stride=2, padding=1, batch_norm=True,
            init_zero_weights=init_zero_weights
        )  # B×(2·cd)×8×8

        # bottleneck with residual connection
        self.resnet_block = ResnetBlock(conv_dim*2, init_zero_weights)
        
        # decoder: 8×8×(conv_dim*2) → 16×16×conv_dim → 32×32×3
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4, stride=2, padding=1, batch_norm=True)
        self.deconv2 = deconv(conv_dim, 3, 4, stride=2, padding=1, batch_norm=False)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.resnet_block(out))
        out = F.relu(self.deconv1(out))
        return torch.tanh(self.deconv2(out))


class DCDiscriminator(nn.Module):
    """PatchGAN‐style discriminator for 32×32 RGB → real/fake"""
    def __init__(self, conv_dim=32):
        super(DCDiscriminator, self).__init__()
        # self.conv1 = conv(3, conv_dim, 4,batch_norm=True)     # 32→16
        # self.conv2 = conv(conv_dim, conv_dim*2, 4,batch_norm=True)  # 16→8
        # self.conv3 = conv(conv_dim*2, conv_dim*4, 4,batch_norm=True) # 8→4
        # self.conv4 = conv(conv_dim*4, 1, 4, stride=1, padding=0, batch_norm=False)        # 4→1
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)     # 32→16
        self.conv2 = conv(conv_dim, int(conv_dim*2), 4, batch_norm=True)  # 16→8
        self.conv3 = conv(int(conv_dim*2), int(conv_dim*4), 4, batch_norm=True) # 8→4
        self.conv4 = conv(int(conv_dim*4), 1, 4, stride=1, padding=0, batch_norm=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out).squeeze()
        return torch.sigmoid(out)

