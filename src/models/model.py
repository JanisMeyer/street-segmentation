import torch
import torch.nn.functional as F

from .encoder import Encoder, VGGEncoder
from .decoder import Decoder, TernausNetDecoder
from ..utils import get_architecture_key
from ..logger import log_info

ARCHITECTURES = {
    "ternausnet_vgg": (VGGEncoder, TernausNetDecoder)
}

class Model(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.mean = torch.nn.Parameter(torch.zeros(3).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(3).view(-1, 1, 1), requires_grad=False)

        log_info("Constructed Model with encoder: %s and decoder %s", self.encoder.__name__, self.decoder.__name__)

    @staticmethod
    def from_config(model_config, data_config):
        architecture_key = get_architecture_key(model_config)
                
        encoder, decoder = ARCHITECTURES.get(architecture_key, (Encoder, Decoder))
        encoder = encoder.from_config(model_config)
        decoder = decoder.from_config(model_config, data_config, encoder)
        return Model(encoder, decoder)

    def forward(self, input):
        encoder_output, encoder_states = self.encoder(input)
        logits = self.decoder(encoder_output, encoder_states)

        result = {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=1),
            "predictions": torch.argmax(logits, dim=1)
        }
        return result