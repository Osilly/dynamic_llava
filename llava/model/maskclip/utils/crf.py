from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pydensecrf import densecrf, utils


class DenseCRF(nn.Module):
    def __init__(
        self,
        max_iter: int = 10,
        pos_w: int = 3,
        pos_xy_std: int = 1,
        bi_w: int = 4,
        bi_xy_std: int = 67,
        bi_rgb_std: int = 3,
        mean: Tuple[int, ...] = (0.485, 0.456, 0.406),
        std: Tuple[int, ...] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.max_iter = max_iter
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.mean = mean
        self.std = std

        self.mean_value = np.float64(np.array(self.mean).reshape(1, -1))
        self.std_value = np.float64(np.array(self.std).reshape(1, -1))

    def _denormalize(self, image):
        image = image.copy().astype(np.float32)

        image = cv2.multiply(image, self.std_value, image)
        image = cv2.add(image, self.mean_value, image)
        image = np.clip(image, 0, 1)
        image = np.uint8(255 * image)

        return image

    def forward_sample(self, output, image):
        output = output.detach().cpu().float()
        image = image.detach().cpu()

        image = image.permute(1, 2, 0).numpy()
        image = self._denormalize(image)
        image = np.ascontiguousarray(image)

        output_probs = F.softmax(output, dim=0).cpu().numpy()

        Nc, H, W = output_probs.shape

        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        d = densecrf.DenseCRF2D(W, H, Nc)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.max_iter)
        Q = np.array(Q).reshape((Nc, H, W))

        return Q

    def forward(self, output, image):
        outputs = [self.forward_sample(output[i], image[i]) for i in range(len(image))]
        output = torch.cat(
            [torch.from_numpy(out).unsqueeze(0) for out in outputs], dim=0
        )

        return output
