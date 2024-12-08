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
        super(ConvLayers, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1)
        self.batch_norm = nn.InstanceNorm2d(out_channel, affine = True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.batch_norm(self.conv(inputs)))
    
class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConv, self).__init__()
        self.upsampling = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv1 = ConvLayers(in_channel, out_channel)

    def forward(self, inputs):
        out = self.upsampling(inputs)
        return self.conv1(out)

class Encode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encode, self).__init__()
        self.mp1 = nn.MaxPool2d(2) 
        self.conv1 = ConvLayers(in_channel, out_channel)
        self.conv2 = ConvLayers(out_channel, out_channel)
    
    def forward(self, input):
        return self.conv2(self.conv1(self.mp1(input)))
    
class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()
        self.up = UpConv(in_channel, out_channel)
        self.conv1 = ConvLayers(in_channel, out_channel)
        self.conv2 = ConvLayers(out_channel, out_channel)
    
    def forward(self, input, con):
        out = self.up(input)
        diffx = con.shape[2] - out.shape[2]
        diffy = con.shape[3] - out.shape[3]
        out = out[:, :, diffx//2 : out.shape[2] - (diffx+1) // 2, diffy//2 : out.shape[3] - (diffy+1)//2]
        out = torch.cat((con, out), dim = 1)
        return self.conv2(self.conv1(out))

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
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
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 1)

        # TODO (student): If you want to use a UNet, you may use this class
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        # TODO (student): If you want to use a UNet, you may use this class
        outputs = self.conv2(self.conv1(outputs))
        out_64 = outputs
        # print(out_64.shape)

        outputs = self.en1(outputs)
        out_128 = outputs

        outputs = self.en2(outputs)
        out_256 = outputs

        outputs = self.en3(outputs)
        out_512 = outputs

        outputs = self.en4(outputs)
        # print(out_1024.shape)
        
        outputs = self.de1(outputs, out_512)
        outputs = self.de2(outputs, out_256)
        outputs = self.de3(outputs, out_128)
        outputs = self.de4(outputs, out_64)
        outputs = self.conv3(outputs)
        # print(inputs.shape, outputs.shape, outputs_up_64.shape)
        outputs = outputs.reshape(batch, -1)
        return outputs
