import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        # self.fc = nn.Linear(512 * 21 * 21, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        # self.fc = nn.Linear(latent_dim, 512 * 21 * 21)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 21 -> 42
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 42 -> 84
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2)  # 84 -> 167
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)  # 167 -> 334

    def forward(self, x):
        # x = self.fc(x)
        # x = x.view(x.size(0), 512, 21, 21)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x

from Unet_EAM import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the Diffusion Process
class DiffusionProcess:
    def __init__(self, beta_start, beta_end, num_timesteps):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        noisy_image = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise
        return noisy_image, noise

    def remove_noise(self, noisy_image, t):
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
        noise_pred = noisy_image / torch.sqrt(alpha_hat_t)
        return noise_pred


# Putting it all together
class DiffusionModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim).to(device)
        self.decoder = Decoder(latent_dim, out_channels).to(device)
        self.unet = UNet(in_channels, in_channels).to(device)
        self.diffusion = DiffusionProcess(beta_start=0.1, beta_end=0.2, num_timesteps=num_timesteps)

    def forward(self, x):
        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        for t in range(self.diffusion.num_timesteps):
            noisy_image, noise = self.diffusion.add_noise(reconstructed, t)
            denoised_image = self.unet(noisy_image)
            reconstructed = self.diffusion.remove_noise(denoised_image, t)


        final_image = self.decoder(self.encoder(reconstructed))
        return final_image


if __name__ == '__main__':
    model = DiffusionModel(in_channels=1, out_channels=1, latent_dim=512, num_timesteps=10).to(device)
    input_tensor = torch.randn((1, 1, 332, 332)).to(device)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
