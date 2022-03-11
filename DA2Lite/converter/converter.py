import torch

from DA2Lite.core.log import get_logger


logger = get_logger(__name__)


class Converter(object):
    def __init__(self, model: torch.nn.Module, save_path: str, input_shape: torch.Tensor = None) -> None:
        self.model = model
        self.input_shape = input_shape
        self.save_path = save_path

    def to_torchscript(self):
        model = torch.jit.trace(self.model, torch.rand(self.input_shape).to(next(self.model.parameters()).device))
        model.save(self.save_path)
        logger.info(f"The trained model is saved in {self.save_path}")
        return model

