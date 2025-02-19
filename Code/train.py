import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import math
import os
import torch.nn as nn
from diffusion1 import DiffusionModel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from SuperResolution import SuperResolutionDataset


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / torch.sqrt(mse))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    writer = SummaryWriter()
    best_val_loss = float('inf')
    best_val_psnr = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将模型包裹在 DataParallel 中以支持多张 GPU
    model = nn.DataParallel(model)

    # 将模型移动到 GPU
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_psnr = 0.0
        ct = 0
        for lr_images, hr_images in train_loader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            ct = ct + len(lr_images)

            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            if ct//1000==0:
                print(ct, 'image number')
                print(outputs.shape,'mdoel shape')
                print(loss.item(), 'model loss')
                print(calculate_psnr(outputs, hr_images), 'train_psnr')
            train_psnr += calculate_psnr(outputs, hr_images)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_psnr = train_psnr / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                noisy_lr_images, noise = model.module.diffusion.add_noise(lr_images, 9)
                outputs = model.module.diffusion.remove_noise(noisy_lr_images, 9)
                loss = criterion(outputs, hr_images)

                running_val_loss += loss.item() * lr_images.size(0)
                val_psnr += calculate_psnr(outputs, hr_images) * lr_images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_psnr = val_psnr / len(val_loader.dataset)

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('PSNR/train', epoch_train_psnr, epoch)
        writer.add_scalar('PSNR/val', epoch_val_psnr, epoch)

        # Save the best models
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_loss_model.pth'))

        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_psnr_model.pth'))

        # Save the last model
        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train PSNR: {epoch_train_psnr:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val PSNR: {epoch_val_psnr:.4f}")

    writer.close()

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 50
    learning_rate = 1e-4
    save_dir = './models'

    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(in_channels=1, out_channels=1, latent_dim=512, num_timesteps=10)

    # 包裹模型以支持多GPU
    model = nn.DataParallel(model).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Hyperparameters
    batch_size = 16
    scale_factor = 4

    # Data preprocessing
    hr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    lr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    # Load custom dataset
    dataset = SuperResolutionDataset(hr_dir='E:/FY_Data/YSL_Code/data/hr', lr_dir='E:/FY_Data/YSL_Code/data/lr_2',
                                     hr_transform=hr_transform, lr_transform=lr_transform)
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir)
