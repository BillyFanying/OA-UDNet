import torch
import os
from model.Unet_EAM import UNet
import torchvision.transforms as transforms
from tqdm import tqdm
def save_image(tensor, images_save_dir, file_name):
    tensor = tensor.cpu().detach().numpy().squeeze()
    tensor = np.array(tensor)
    # tensor = tensor * 2
    tensor = (tensor * np.array([0.25]) + np.array([0.5]))*255
    tensor = tensor.astype(np.uint8)
    # print(tensor)
    img = Image.fromarray(tensor)
    # img.show()
    img.save(os.path.join(images_save_dir, file_name))

'''
origingal_image:hr
generated_image:out_img
'''
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import numpy as np

def calculate_metrics(original_image, generated_image):
    # 确保图像数据类型为 float
    original_image = np.array(original_image)
    generated_image = np.array(generated_image)
    # 计算峰值信噪比（PSNR）
    psnr = peak_signal_noise_ratio(original_image, generated_image)

    # 计算结构相似性指数（SSIM），手动指定数据范围为 0 - 255（假设图像是 8 位灰度图像）
    ssim = structural_similarity(original_image, generated_image, data_range=255, multichannel=False)
    return psnr, ssim
def zhibiaopinggu():
    out_img_path = ''
    hr_img_path = ''
    out_img = np.array(Image.open(out_img_path).convert("L"))
    hr_img = np.array(Image.open(hr_img_path).convert("L"))

    psnr_value, ssim_value = calculate_metrics(out_img, hr_img)
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")


def Generate_test_images(model_pth,lr_img_path,images_save_dir):
    gpu = 1
    torch.cuda.set_device(gpu)
    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth))
    lr_img_path = lr_img_path
    images_save_dir = images_save_dir

    lr_image = Image.open(lr_img_path).convert("L")
    lr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    lr_image = lr_transform(lr_image)
    lr_image = np.array(lr_image).astype(np.float32)
    lr_image = lr_image / np.max(lr_image)
    lr_image = torch.from_numpy(lr_image)
    lr_image = lr_image.unsqueeze(0).to(device)


    print(lr_image.shape)
    model.eval()
    with torch.no_grad():
        outputs = model(lr_image)
        save_image(outputs,images_save_dir, file_name='output1.png')


    print("end-----")


def Generate_batch_test_images(model_pth,dir,images_save_dir):

    gpu = 1
    torch.cuda.set_device(gpu)
    # Model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_pth))
    lr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])
    images_save_dir = images_save_dir
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)
    lr_img_dir = os.listdir(dir)
    tq = tqdm(lr_img_dir, total=len(lr_img_dir))
    for li in lr_img_dir:
        lr_img_path = os.path.join(dir, li)
        lr_image = Image.open(lr_img_path).convert("L")
        lr_image = np.array(lr_image).astype(np.float32)
        lr_image = lr_image / np.max(lr_image)
        lr_image = lr_transform(lr_image)
        lr_image = lr_image.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(lr_image)
            save_image(outputs, images_save_dir, file_name=li)
        tq.update(1)


    print("end-----")


root = ''
save_root = ''
fbls = ['']
bws = ['']
for fbl in fbls:
    for name in bws:
        dir = root+fbl+'/'+name
        images_save_dir = save_root+fbl+'/'+name
        model_path = 'models'+fbl+'_over/'+fbl+'_overbest_val_psnr_model.pth'
        Generate_batch_test_images(model_path,dir,images_save_dir)

