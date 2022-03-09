import torch
import torch.nn.functional as F

class Attention(torch.nn.Module):
    def __init__(self, channels_enc, channels_dec):
        super(Attention, self).__init__()
        self.channels_enc = channels_enc
        self.channels_dec = channels_dec

    @staticmethod
    def from_config(model_config):
        return ARCHITECTURES.get(model_config.get("attention", "").lower(), Attention)

    def forward(self, input_enc, input_dec):
        return input_dec

class CopyAttention(Attention):
    def forward(self, input_enc, input_dec):
        return torch.cat([input_enc, input_dec], dim=1)


ARCHITECTURES = {
    "copy": CopyAttention
}