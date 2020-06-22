import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from typing import Tuple


class Dummy(VisionDataset):
    def __init__(self, root, loader=default_loader, extensions=None, transform=None,
                 target_transform=None):
        super(Dummy, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = loader
        self.extensions = extensions

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = torch.randn(3, 224, 224)
        mask = torch.zeros(1, 224, 224).long()
        return image, mask

    def __len__(self):
        return 100_500
