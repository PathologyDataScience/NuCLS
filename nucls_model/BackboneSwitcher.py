# Modified from Matthias Hamacher
# https://github.com/CancerDataScience/CellDetectionCenterpoint/

# from torchvision import models
from torch.nn import Sequential
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class BackboneSwitcher:

    @staticmethod
    def feature_nets(fun, pretrained):
        """
        Get the convolution layers of a network architecture, with all
        convolution layers in a torch.nn.Sequential or Module element
        named features
        :param fun: a function has a boolean as input.
            This function should return the
            network architecture (Pre-trained if the boolean is True)
        :param pretrained: boolean input for the function fun.
        :return: convolution layers of the network architecture
        """
        return fun(pretrained).features

    @staticmethod
    def resnet(fun, pretrained):
        """
        Does the same as feature_nets, but for resnet architectures.
        :param fun: see feature_nets
        :param pretrained: see feature_nets
        :return: see feature_nets
        """
        resnet = fun(pretrained)
        return Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    @staticmethod
    def resnet_fpn(backbone_name: str, pretrained, trainable_layers=3):
        """
        Does the same as resnet, but with a feature pyramidal network component
        :param backbone_name: name of resnet backbone (eg resnet50)
        :param pretrained: see feature_nets
        :return: see feature_nets
        """
        return resnet_fpn_backbone(
            backbone_name, pretrained, trainable_layers=trainable_layers)
