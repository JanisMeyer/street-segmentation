import torch
from torchvision.models.vgg import vgg19_bn

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.out_channels = []
    
    @staticmethod
    def from_config(model_config):
        return Encoder()

class VGGEncoder(Encoder):
    def __init__(self, pretrained=False):
        super(VGGEncoder, self).__init__()
        backbone = vgg19_bn(pretrained=pretrained).features

        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(*backbone[:6]),
            torch.nn.Sequential(*backbone[7:13]),
            torch.nn.Sequential(*backbone[14:26]),
            torch.nn.Sequential(*backbone[27:39]),
            torch.nn.Sequential(*backbone[40:52])
        ])
        
        self.pooling_layers = torch.nn.ModuleList([
            backbone[6],
            backbone[13],
            backbone[26],
            backbone[39],
            backbone[52]
        ])
        self.out_channels = [64, 128, 256, 512, 512, 512]

    @staticmethod
    def from_config(model_config):
        return VGGEncoder(pretrained=model_config.get("pretrained", False))

    def forward(self, x):
        encoder_states = []
        for block, pooling_layer in zip(self.blocks, self.pooling_layers):
            _x = block(x)
            encoder_states.append(_x)
            x = pooling_layer(_x)
        return x, encoder_states