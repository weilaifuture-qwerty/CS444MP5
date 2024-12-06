import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, latent_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_size=32, hidden_size=128, output_size=2):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.decoder(x)
    
class ConvLayers(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.batch_norm(self.conv(inputs)))
    
class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor = 2)
        self.conv1 = ConvLayers(in_channel, out_channel)

    def forward(self, inputs):
        out = self.upsampling(inputs)
        return self.conv1(out)

class Encode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mp1 = nn.MaxPool2d(2, stride = 2) 
        self.conv1 = ConvLayers(in_channel, out_channel)
        self.conv2 = ConvLayers(out_channel, out_channel)
    
    def forward(self, input):
        return self.conv2(self.conv1(self.mp1(input)))
    
class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = UpConv(in_channel, out_channel)
        self.conv1 = ConvLayers(in_channel, out_channel)
        self.conv2 = ConvLayers(out_channel, out_channel)
    
    def forward(self, input, con):
        out = self.up(input)
        out = torch.cat((con, out), dim = 1)
        return self.conv2(self.conv1(out))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayers(1, 64)
        self.conv2 = ConvLayers(64, 64)
        self.en1 = Encode(64, 128)
        self.en2 = Encode(128, 256)
        self.en3 = Encode(256, 512)
        self.en4 = Encode(512, 1024)

        self.de1 = Decode(1024, 512)
        self.de2 = Decode(512, 256)
        self.de3 = Decode(256, 128)
        self.de4 = Decode(128, 64)
        self.conv3 = ConvLayers(64, 1)

        # TODO (student): If you want to use a UNet, you may use this class
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        # TODO (student): If you want to use a UNet, you may use this class
        out_64 = self.conv2(self.conv1(outputs))
        # print(out_64.shape)
        out_128 = self.en1(out_64)
        out_256 = self.en2(out_128)
        out_512 = self.en3(out_256)
        out_1024 = self.en4(out_512)
        # print(out_1024.shape)
        
        out_up_512 = self.de1(out_1024, out_512)
        out_up_256 = self.de2(out_up_512, out_256)
        out_up_128 = self.de3(out_up_256, out_128)
        out_up_64 = self.de4(out_up_128, out_64)
        outputs = self.conv3(out_up_64)
        # print(inputs.shape, outputs.shape, outputs_up_64.shape)
        outputs = outputs.reshape(batch, -1)
        return outputs
