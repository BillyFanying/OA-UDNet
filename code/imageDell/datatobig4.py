from PIL import Image
import os

def enlarge_image(input_path, output_path, scale_factor):
    # 打开图片
    with Image.open(input_path) as img:
        # 获取原始图片的尺寸
        original_size = img.size
        print(f"Original size: {original_size}")

        # 计算放大后的尺寸
        # new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
        new_size = (1328,1328)
        print(f"New size: {new_size}")

        # 使用resize方法放大图片
        enlarged_img = img.resize(new_size)

        # 保存放大后的图片
        enlarged_img.save(output_path)
        print(f"Enlarged image saved to {output_path}")


# 示例使用
# output_image_path = r'E:\FY_Data\Data_Selected_Final\Primarytest4\32\32_test\32_2024-07-28 - Study_Scan_8_save_Signal_Pos_23_91.00_Wave_1_660.00.mat.png'  # 替换为你的输出图片路径
scale_factor = 4  # 放大四倍

inputFilepath = r'E:\FY_Data\Data_Selected_Final\Deepmb_Result\unet_result\unet'
outputFilepath = r'E:\FY_Data\Data_Selected_Final\Deepmb_Result\unet_result\unet1328'
sizeFilenames = os.listdir(inputFilepath)
for sizeFilename in sizeFilenames:
    sizePath = os.path.join(inputFilepath, sizeFilename)
    inFilenames = os.listdir(sizePath)
    for inFilename in inFilenames:
        # if '_test' in inFilename: #E:\FY_Data\Data_Selected_Final\DDPM这个路径下要打开注释
            inFilepath = os.path.join(sizePath, inFilename)
            imageNames = os.listdir(inFilepath)
            for imageName in imageNames:
                input_image_path = os.path.join(inFilepath, imageName)
                output_image_File =outputFilepath+'\\'+sizeFilename+'\\'+inFilename
                if not os.path.exists(output_image_File):
                    os.makedirs(output_image_File)
                output_image_path = os.path.join(output_image_File, imageName)
                enlarge_image(input_image_path, output_image_path, scale_factor)
    print(sizeFilename+'------------------------------------------------')

