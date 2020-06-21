import torch
import torch.nn.functional as F

from torch import Tensor


def image_dx(x: Tensor, pad=False) -> Tensor:
    dx = x[:, :, :, :-1] - x[:, :, :, 1:]
    if pad:
        dx = F.pad(dx, [0, 1])
    else:
        # throw away bottom pixel row (y) to get a square matrix
        dx = dx[:, :, :-1]
    return dx


def image_dy(x: Tensor, pad=False) -> Tensor:
    dx = x[:, :, :-1] - x[:, :, 1:]
    if pad:
        dx = F.pad(dx, [0, 0, 0, 1])
    else:
        # throw away right-most pixel column (x) to get a square matrix
        dx = dx[:, :, :, :-1]
    return dx


def consistency_loss(image: Tensor, mask: Tensor, eps=1e-7, method="diff") -> Tensor:
    assert mask.size(1) == 1
    if image.size(1) > 1:
        # not sure if the real grayscale image is needed,
        # let's start with a simple mean over channels dim
        image = image.mean(1, keepdim=True)

    if method == "diff":
        I_x = image_dx(image)
        I_y = image_dy(image)
        M_x = image_dx(mask)
        M_y = image_dy(mask)

    elif method == "sobel":
        K_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]).float().div_(8)[None, None]
        K_y = K_x.transpose(-1, -2).contiguous()

        I_x = F.conv2d(image, K_x)
        I_y = F.conv2d(image, K_y)
        M_x = F.conv2d(mask, K_x)
        M_y = F.conv2d(mask, K_y)

    else:
        raise AttributeError("Unknown method: {}".format(method))

    # mask grad magnitude
    M = torch.sqrt(M_x ** 2 + M_y ** 2)

    # grad disagreement term
    D = torch.pow(I_x * M_x + I_y * M_y, 2)
    D = torch.clamp_min(torch.ones_like(D) - D, 0)
    return torch.sum(M * D) / (torch.sum(M) + eps)
