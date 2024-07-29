import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import utils

from .palette import available_palette, get_palette


def draw_segmentation(img, out, dataset=None, num_classes=None, random_state=42):
    if dataset and dataset in available_palette():
        colors = list(map(tuple, get_palette(dataset)))
    elif num_classes is not None:
        np.random.seed(random_state)
        colors = np.random.randint(0, 255, size=(num_classes, 3))
        colors = list(map(tuple, colors))
    else:
        raise ValueError("`dataset` or `num_classes` should be specified")

    out = F.interpolate(
        out, size=(img.size[1], img.size[0]), mode="bilinear", align_corners=False
    )
    out = out.argmax(dim=1)[0]

    img_tensor = TF.to_tensor(img).multiply(255).to(torch.uint8)
    outs = torch.zeros([len(colors), *out.shape], dtype=torch.bool)

    for i in range(len(colors)):
        outs[i] = torch.where(out == i, True, False)

    masked_img = utils.draw_segmentation_masks(img_tensor, outs, colors=colors).permute(
        1, 2, 0
    )
    img_grid = [img_tensor.permute(1, 2, 0), masked_img]
    return img_grid
