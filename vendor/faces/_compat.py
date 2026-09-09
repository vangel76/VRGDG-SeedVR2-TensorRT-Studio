"""Minimal stand-ins for the basicsr helpers the vendored archs import."""

from __future__ import annotations

import logging

import torch
from torch import nn


class _Registry:
    def register(self, obj=None, **kwargs):
        if obj is None:
            return lambda inner: inner
        return obj


ARCH_REGISTRY = _Registry()


def get_root_logger(*args, **kwargs) -> logging.Logger:
    return logging.getLogger("vendor.faces")


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Kaiming-normal init used by BasicSR; identical semantics to basicsr.archs.arch_util."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
