import torch
import torch.nn as nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)


class ProxyFeatureExtractor(nn.Module):
    """
    Extract intermediate FPN features from a pretrained
    Mask R-CNN ResNet-50 FPN model.

    The NN-VVC paper uses features from a pretrained
    Mask R-CNN ResNet-50 FPN as a task-agnostic proxy
    representation.

    This implementation is a prototype and does not claim
    to reproduce the paper's exact multi-layer proxy loss.
    """

    def __init__(self):
        super().__init__()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
        )

        # Proxy network is pretrained and kept fixed.
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        """
        Args:
            images:
                Tensor of shape [B, 3, H, W], expected in [0, 1].

        Returns:
            Dictionary of intermediate FPN feature maps.
        """

        image_list, _ = self.model.transform(images)

        features = self.model.backbone(image_list.tensors)

        return features


class ProxyFeatureLoss(nn.Module):
    """
    Compare proxy features of an original image and its
    reconstructed version.

    The loss is computed as the average MSE across the
    available FPN feature levels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, original_features, reconstructed_features):
        losses = []

        for key in original_features.keys():
            original = original_features[key]
            reconstructed = reconstructed_features[key]

            losses.append(
                torch.mean((original - reconstructed) ** 2)
            )

        return torch.stack(losses).mean()