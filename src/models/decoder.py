import torch
from .attention import Attention, CopyAttention

class Decoder(torch.nn.Module):
    @staticmethod
    def from_config(model_config, data_config, encoder=None):
        return Decoder()

class TernausNetDecoder(Decoder):
    def __init__(self, num_classes, in_channels, attention_cls=CopyAttention):
        super(TernausNetDecoder, self).__init__()
        
        num_channels = [(in_channels[5], 256), (256 + in_channels[4], 256), (256 + in_channels[3], 128), (128 + in_channels[2], 64), (64 + in_channels[1], 32)]
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv2d(_in, _out, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(_out),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)) for _in, _out in num_channels])
        self.attention = torch.nn.ModuleList([attention_cls(_enc, _dec) for (_, _dec), _enc in zip(num_channels, in_channels[4::-1])])
        
        self.last_conv = torch.nn.Sequential(
            torch.nn.Conv2d(32 + in_channels[0], 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU())
        self.output = torch.nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def from_config(model_config, data_config, encoder=None):
        return TernausNetDecoder(len(data_config.get("labels", [])), encoder.out_channels, attention_cls=Attention.from_config(model_config))

    def forward(self, x, encoder_states):
        for block, attention, encoder_state in zip(self.blocks, self.attention, encoder_states[::-1]):
            x = block(x)
            x = attention(encoder_state, x)
        x = self.last_conv(x)
        x = self.output(x)
        return x
        