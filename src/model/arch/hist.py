import torch


def get_gray(img):
    r = img[:, 0, ...]
    g = img[:, 1, ...]
    b = img[:, 2, ...]
    return (0.299 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def pack_tensor(x, n_bins):
    # pack tensor: transform gt_hist.shape [n_bins, bs, c, h, w] -> [bs*c, b_bins, h, w]
    # merge dim 1 (bs) and dim 2 (channel).
    return x.reshape(n_bins, -1, *x.shape[-2:]).permute(1, 0, 2, 3)


def get_hist(img, n_bins, grayscale=False):
    """
    Given a img (shape: bs, c, h, w),
    return the SOFT histogram map (shape: n_bins, bs, c, h, w)
                                or (shape: n_bins, bs, h, w) when grayscale=True.
    """
    if grayscale:
        img = get_gray(img)
    return torch.stack([
        torch.nn.functional.relu(1 - torch.abs(img - (2 * b - 1) / float(2 * n_bins)) * float(n_bins))
        for b in range(1, n_bins + 1)
    ])


def get_hist_conv(n_bins, kernel_size=2, train=False):
    """
    Return a conv kernel.
    The kernel is used to apply on the histogram map, shrinking the scale of the hist-map.
    """
    conv = torch.nn.Conv2d(n_bins, n_bins, kernel_size, kernel_size, bias=False, groups=1)
    conv.weight.data.zero_()
    for i in range(conv.weight.shape[1]):
        alpha = kernel_size ** 2
        #         alpha = 1
        conv.weight.data[i, i, ...] = torch.ones(kernel_size, kernel_size) / alpha
    if not train:
        conv.requires_grad_(False)
    return conv
