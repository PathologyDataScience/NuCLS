# Modified from Matthias Hamacher
# https://github.com/CancerDataScience/CellDetectionCenterpoint/
# See: https://pytorch.org/docs/stable/torchvision/models.html

import sys
from torch.nn import Module
from torchvision import models
BASEPATH = "/home/mtageld/Desktop/cTME/"
sys.path.insert(0, BASEPATH)
from nucl_model.BackboneSwitcher import BackboneSwitcher as bbs


class FeatureExtractor(Module):
    def __init__(self, nntype: str, pretrained: bool = True):
        """
        class with feature extraction part of bbx regression
        :param nntype: string architecture of neural network model
        :param pretrained: boolean whether the layers should be initialized
            True -> layers are loaded pre-trained on ImageNet
            False -> layers are randomly initialized
        """
        super(FeatureExtractor, self).__init__()
        self.layers, self.out_channels = self.get_layers(nntype, pretrained)

    def forward(self, X):
        return self.layers(X)

    def get_layers(self, nntype, pretrained):
        """
        methods loads model type and extracts the fully convolutional layers
        :param nntype: string architecture of neural network model
        :param pretrained: boolean whether the layers should be initialized
            True -> layers are loaded pre-trained on ImageNet
            False -> layers are randomly initialized
        :return: fully convolutional layers, and the number of output channels
        """
        model_switcher = {
            'alexnet': (bbs.feature_nets, models.alexnet, 256),

            'vgg11': (bbs.feature_nets, models.vgg11, 512),
            'vgg13': (bbs.feature_nets, models.vgg13, 512),
            'vgg16': (bbs.feature_nets, models.vgg16, 512),
            'vgg19': (bbs.feature_nets, models.vgg19, 512),
            'vgg11bn': (bbs.feature_nets, models.vgg11_bn, 512),
            'vgg13bn': (bbs.feature_nets, models.vgg13_bn, 512),
            'vgg16bn': (bbs.feature_nets, models.vgg16_bn, 512),
            'vgg19bn': (bbs.feature_nets, models.vgg19_bn, 512),

            'resnet18': (bbs.resnet, models.resnet18, 512),
            'resnet34': (bbs.resnet, models.resnet34, 512),
            'resnet50': (bbs.resnet, models.resnet50, 2048),
            'resnet101': (bbs.resnet, models.resnet101, 2048),
            'resnet152': (bbs.resnet, models.resnet152, 512),

            # # mTODO(?): FPN runs until the RPN part then has a size
            # #  compatibility issue for the anchor propodal numbers. I can
            # #  debug this but I have other high priority tasks
            # # feature pyramidal network
            # 'resnet18-fpn': (bbs.resnet_fpn, 'resnet18', 256),
            # 'resnet50-fpn': (bbs.resnet_fpn, 'resnet50', 256),

            # 'googlenet': None,
            'mobilenet': (bbs.feature_nets, models.mobilenet_v2, 1280),

        }
        case = model_switcher.get(nntype)
        if case is None:
            raise NNTypeError(
                "{} is no neural network architecture implemented in this " +
                "class.\n The implemented architectures are: {}.".format(
                    str(model_switcher.keys())[11:-2]))
        cal, fun, in_dim = case
        return cal(fun, pretrained), in_dim


class NNTypeError(Exception):
    pass
