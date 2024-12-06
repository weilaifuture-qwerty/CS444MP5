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
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.conv1(inputs))
    
class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.upsampling(inputs)
        return self.relu(self.conv1(inputs))

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = ConvLayers(in_channel, out_channel, 3, 1, 1)
        self.conv2 = ConvLayers(out_channel, out_channel, 3, 1, 1)
    
    def forward(self, input):
        return self.conv2(self.conv1(input))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(1, 64)

        self.mp1 = nn.MaxPool2d(2, stride = 2) # 64 * 284 * 284
        self.conv2 = Conv(64, 128)

        self.mp2 = nn.MaxPool2d(2, stride = 2) # 128 * 140 * 140
        self.conv3 = Conv(128, 256)

        self.mp3 = nn.MaxPool2d(2, stride = 2) # 256 * 68 * 68
        self.conv4 = Conv(256, 512)

        self.mp4 = nn.MaxPool2d(2, stride = 2) # 512 * 32 * 32
        self.conv5 = Conv(512, 1024)

        self.up1 = UpConv(1024, 512, 3, 1, 1)
        self.conv6 = Conv(1024, 512)

        self.up2 = UpConv(512, 256, 3, 1, 1)
        self.conv7 = Conv(512, 256)

        self.up3 = UpConv(256, 128, 3, 1, 1)
        self.conv8 = Conv(256, 128)

        self.up4 = UpConv(128, 64, 3, 1, 1)
        self.conv9 = Conv(128, 64)

        self.conv10 = ConvLayers(64, 1, 1, 1, 0)

        # TODO (student): If you want to use a UNet, you may use this class
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        # TODO (student): If you want to use a UNet, you may use this class
        outputs_64 = self.conv1(outputs)

        outputs_128 = self.mp1(outputs_64)
        outputs_128 = self.conv2(outputs_128)

        outputs_256 = self.mp2(outputs_128)
        outputs_256 = self.conv3(outputs_256)

        outputs_512 = self.mp3(outputs_256)
        outputs_512 = self.conv4(outputs_512)

        outputs_1024 = self.mp4(outputs_512)
        outputs_1024 = self.conv5(outputs_1024)

        outputs_up_512 = self.up1(outputs_1024)
        outputs_up_512 = torch.cat((outputs_up_512, outputs_512), dim = 1)
        outputs_up_512 = self.conv6(outputs_up_512)

        outputs_up_256 = self.up2(outputs_up_512)
        outputs_up_256 = torch.cat((outputs_up_256, outputs_256), dim = 1)
        outputs_up_256 = self.conv7(outputs_up_256)

        outputs_up_128 = self.up3(outputs_up_256)
        outputs_up_128 = torch.cat((outputs_up_128, outputs_128), dim = 1)
        outputs_up_128 = self.conv8(outputs_up_128)

        outputs_up_64 = self.up4(outputs_up_128)
        outputs_up_64 = torch.cat((outputs_up_64, outputs_64), dim = 1)
        outputs_up_64 = self.conv9(outputs_up_64)
        outputs = self.conv10(outputs_up_64)

        # print(inputs.shape, outputs.shape, outputs_up_64.shape)
        outputs = outputs.reshape(batch, -1)
        return outputs
