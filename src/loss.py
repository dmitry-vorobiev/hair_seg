import torch
import torch.nn.functional as F

from torch import Tensor
from torch.jit import trace


def diff(x: Tensor, dim: int, pad=True) -> Tensor:
    r"""
    x.shape: (8, 3, 224, 224)
    dim: -1

    mask0: [:, :, :, 1:]
    mask1: [:, :, :, :-1] <- shifted by 1 pixel along corresponding dim
    """
    first_dims = list(map(slice, x.shape[:dim]))
    mask0 = [*first_dims, slice(1, x.size(dim))]
    mask1 = [*first_dims, slice(0, -1)]
    d = x[mask0] - x[mask1]

    if pad:
        pad_amount = [0, 0] * int(x.ndim)
        pad_amount[dim * 2 + 1] = 1
        d = F.pad(d, list(reversed(pad_amount)))

    return d


def image_dx(x: Tensor) -> Tensor:
    dx = x[:, :, :, :-1] - x[:, :, :, 1:]
    # throw away bottom pixel row to get a square matrix
    dx = dx[:, :, :-1]
    return dx


def image_dy(x: Tensor) -> Tensor:
    dx = x[:, :, :-1] - x[:, :, 1:]
    # throw away right-most pixel column to get a square matrix
    dx = dx[:, :, :, :-1]
    return dx


def conv_sep(x: Tensor, u: Tensor, v: Tensor) -> Tensor:
    x = F.conv2d(x, u[None, None, :, None])
    x = F.conv2d(x, v[None, None, None, :])
    return x


def simple_norm(x: Tensor, eps=1e-7) -> Tensor:
    norm = torch.abs(x) + eps
    return x / norm


dummy_img = torch.randn(2, 1, 224, 224)
image_dx = trace(image_dx, (dummy_img,))
image_dy = trace(image_dy, (dummy_img,))
conv_sep = trace(conv_sep, (dummy_img, torch.rand(7), torch.rand(7)))
simple_norm = trace(simple_norm, (dummy_img,))


def consistency_loss(image: Tensor, mask: Tensor, eps=1e-7, method="diff",
                     norm_grads=True) -> Tensor:
    assert mask.size(1) == 1
    if image.size(1) > 1:
        # not sure if the real grayscale image is needed,
        # let's start with a simple mean over channels dim
        image = image.mean(1, keepdim=True)

    device = image.device

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
        ], device=device).float().div_(8)[None, None]
        K_y = K_x.transpose(-1, -2).contiguous()

        I_x = F.conv2d(image, K_x)
        I_y = F.conv2d(image, K_y)
        M_x = F.conv2d(mask, K_x)
        M_y = F.conv2d(mask, K_y)

    elif method.startswith("farid"):
        #  H. Farid and E. P. Simoncelli, Differentiation of discrete multi-dimensional signals,
        #  IEEE Trans Image Processing, vol.13(4), pp. 496--508, Apr 2004.
        size = method[-1]
        if size == str(5):
            k = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
            d = [0.104550, 0.292315, 0.000000, -0.292315, -0.104550]
        elif size == str(7):
            k = [0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711]
            d = [0.018708, 0.125376, 0.193091, 0.000000, -0.193091, -0.125376, -0.018708]
        else:
            raise AttributeError("Unsupported: {}".format(method))

        k = torch.tensor(k, device=device)
        d = torch.tensor(d, device=device)

        I_x = conv_sep(image, k, d)
        I_y = conv_sep(image, d, k)
        M_x = conv_sep(mask, k, d)
        M_y = conv_sep(mask, d, k)

    else:
        raise AttributeError("Unknown method: {}".format(method))

    # mask grad magnitude
    M = torch.sqrt(M_x ** 2 + M_y ** 2)

    if norm_grads:
        I_x = simple_norm(I_x)
        I_y = simple_norm(I_y)
        M_x = simple_norm(M_x)
        M_y = simple_norm(M_y)

    # grad disagreement term
    D = torch.pow(I_x * M_x + I_y * M_y, 2)
    D = torch.clamp_min(torch.ones_like(D) - D, 0)
    return torch.sum(M * D) / (torch.sum(M) + eps)
