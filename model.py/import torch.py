import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple

class ReflectionConvBlock(nn.Module):
    """
    Convolutional block with Reflection Padding to reduce border artifacts.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ReflectionConvBlock, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    """
    Improved Residual Block with Reflection Padding and Instance Normalization.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ReflectionConvBlock(channels, channels, kernel_size=3)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ReflectionConvBlock(channels, channels, kernel_size=3)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvBlock(nn.Module):
    """
    Upsampling block using Nearest Neighbor interpolation followed by convolution
    to avoid checkerboard artifacts common in Transposed Convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=None):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class ImprovedTransformNet(nn.Module):
    """
    Optimized Transform Network for Fast Neural Style Transfer.
    Improvements:
    - Reflection Padding instead of Zero Padding.
    - More Residual Blocks (8 instead of 5).
    - Affine Instance Normalization.
    - Scaled output instead of raw Sigmoid.
    """
    def __init__(self):
        super(ImprovedTransformNet, self).__init__()
        # Initial Layers
        self.conv1 = ReflectionConvBlock(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ReflectionConvBlock(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ReflectionConvBlock(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual Layers (Increased depth)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(8)])

        # Upsampling Layers
        self.up1 = UpsampleConvBlock(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.up2 = UpsampleConvBlock(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.conv4 = ReflectionConvBlock(32, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.res_blocks(out)
        out = self.relu(self.in4(self.up1(out)))
        out = self.relu(self.in5(self.up2(out)))
        out = self.conv4(out)
        return out

class VGG19Extractor(nn.Module):
    """
    Multi-layer VGG19 Feature Extractor.
    Standard NST uses relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 for style.
    """
    def __init__(self):
        super(VGG19Extractor, self).__init__()
        vgg_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_features[x])
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"])
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1)
        return out

def gram_matrix(y):
    """
    Computes the Gram Matrix for style representation.
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

# Example Loss Calculation with Multi-Layer Support
def calculate_loss(output, content_y, style_grams, vgg, content_weight=1e5, style_weight=1e10):
    features_y = vgg(output)
    features_content = vgg(content_y)
    
    # Content loss (usually relu2_2 or relu3_3)
    content_loss = content_weight * nn.functional.mse_loss(features_y.relu2_1, features_content.relu2_1)
    
    # Style loss (Multi-layer)
    style_loss = 0
    for ft_y, gm_s in zip(features_y, style_grams):
        gm_y = gram_matrix(ft_y)
        style_loss += nn.functional.mse_loss(gm_y, gm_s[:output.size(0), :, :])
    style_loss *= style_weight
    
    return content_loss + style_loss
