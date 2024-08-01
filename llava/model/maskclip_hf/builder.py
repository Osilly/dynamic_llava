from .maskclip import MaskCLIP


def build_maskclip(maskclip, maskclip_cfg, **kwargs):
    return MaskCLIP(maskclip, maskclip_cfg, **kwargs)
