import torch
import torch.nn as nn
import torch.nn.functional as F
from sobelCovn import SobelConv2d
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SobelLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(SobelLayer, self).__init__()
        self.esa = nn.Sequential(
            SobelConv2d(in_channels, out_channels),
        )

    def forward(self, x):
        return self.esa(x)

class Fc(nn.Module):
    def __init__(self, c, h, w):
        super(Fc, self).__init__()
        self.fc_loc = nn.Sequential(
            nn.Linear(c * h * w, 32),  # Adjusted the input size
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        return self.fc_loc(x)

class EAS(nn.Module):
    def __init__(self, in_channels):
        super(EAS, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = SobelLayer(xs.shape[1],xs.shape[1])(xs)
        sz = xs
        xs = xs.view(-1, 10 * xs.shape[2] * xs.shape[3])  # Adjusted the flattened size
        theta =Fc(sz.shape[1], sz.shape[2], sz.shape[3]).to(device)(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid,align_corners=True)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoding path
        self.enc1 = nn.Sequential(
            EAS(in_channels),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2,padding=1)

        self.enc2 = nn.Sequential(
            EAS(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            EAS(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = nn.Sequential(
            EAS(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoding path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            EAS(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            EAS(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            EAS(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            EAS(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        center = self.center(self.pool4(enc4))

        # Decoding path
        dec4 = self.up4(center)
        dec4 = self.crop_and_concat(enc4, dec4)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = self.crop_and_concat(enc3, dec3)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = self.crop_and_concat(enc2, dec2)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = self.crop_and_concat(enc1, dec1)
        dec1 = self.dec1(dec1)

        final = self.final(dec1)

        return final

    def crop_and_concat(self, enc, dec):
        """
        Center crop `enc` to have the same spatial size as `dec` and concatenate them along the channel axis.
        """
        _, _, h, w = enc.size()
        _, _, h2, w2 = dec.size()
        dh = (h - h2) // 2
        dw = (w - w2) // 2
        enc = enc[:, :, dh:dh+h2, dw:dw+w2]
        return torch.cat([enc, dec], dim=1)


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=3).to(device)
    input_tensor = torch.randn((1, 3, 336, 336)).to(device)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
