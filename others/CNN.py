import torch.nn as nn
from torch.nn import functional as nnf

class CNNEmbedder(nn.Module):
    """The embedder part of the `CNNet` classifier."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, data):
        """Do the forward pass.
        Parameters
        ----------
        data : torch_geometric.data.Batch | torch.Tensor
            A batch of the MorphologyDataset dataset.
        Returns
        -------
        x : torch.Tensor
            The embedding of the persistence image. The dimension of
            the tensor is (n_batch, 3 * (image_size // 4)**2).
        """
        # Note: case in else section can be removed as soon as they implement
        # the feature, see https://github.com/pytorch/captum/issues/494
        if hasattr(data, "image"):
            x = data.image
        else:
            # captum tensors for XAI, see `xai.model_attributions`
            x = data
        x = self.conv1(x)
        x = nnf.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nnf.relu(x)
        x = self.pool2(x)
        x = x.flatten(start_dim=1)

        return x
    
class CNNet(nn.Module):
    """Convolutional net for classifying persistence images.
    The provided persistence images should be square greyscale
    images and can be obtained from persistence diagrams by
    applying Gaussian KDE.
    Parameters
    ----------
    n_classes : int
        The number of classes.
    image_size : int
        The width or height of the input images. The images
        are assumed to be square.
    bn : bool
        If true then 1d batch normalization and a relu are
        applied to the flattened embeddings before the FC
        layer.
    """

    def __init__(self, n_classes, image_size=100, bn=False):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.bn = bn

        # network backbone - feature extractor
        self.feature_extractor = CNNEmbedder()

        # network head - classifier
        n_out_features = 3 * (self.image_size // 4) ** 2
        if bn:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features=n_out_features),
                nn.ReLU(),
                nn.Linear(in_features=n_out_features, out_features=self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=n_out_features, out_features=self.n_classes),
            )

    def forward(self, data):
        """Do the forward pass.
        Parameters
        ----------
        data : torch_geometric.data.Batch
            A batch of MorphologyDataset dataset.
        Returns
        -------
        logits : torch.Tensor
            The predicted logits of shape (n_images, n_classes)
        """
        x = self.feature_extractor(data)

        x = self.classifier(x)
        return x
