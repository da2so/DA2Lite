import pytest
import torch
import os

from DA2Lite.network import resnet18
from DA2Lite.converter.converter import Converter


def test_Converter():
    # Arrange
    model = resnet18()
    save_path = "/tmp/test.pt"

    # Action
    cv = Converter(model, save_path, (1, 3, 32, 32))
    cv.to_torchscript()

    # Assert
    model = torch.jit.load(save_path)
    out = model(torch.rand(16, 3, 32, 32))
    assert list(out.size()) == [16, 10]

    os.remove(save_path)