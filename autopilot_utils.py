import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np

# 定义均值和标准差  Mean and standard deviation values for normalization
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()       
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

# 定义图像预处理函数 Preprocesses the image by converting it to a PyTorch tensor and normalizing it
def preprocess_image(image):
    image = PIL.Image.fromarray(image)                          # 将图像转换为PIL图像对象 Convert image array to PIL.Image object
    image = transforms.functional.to_tensor(image).cuda()       # 将图像转换为Tensor，并移至GPU # Convert PIL.Image to PyTorch tensor
    image.sub_(mean[:, None, None]).div_(std[:, None, None])    # 对图像进行归一化处理 Normalize the image tensor
    return image[None, ...]                                     # 在第0维度上增加一个维度，以匹配模型的输入要求  Add a batch dimension to the image tensor

# 定义中心裁剪函数，使图像变为正方形 Performs a center crop to make the frame square
def center_crop_square(frame):
    src_height, src_width, _ = frame.shape      # 获取原始图像的尺寸 Gets the dimensions of the original image
    src_aspect_ratio = src_width/src_height     # 计算原始图像的宽高比和填充值 Calculate the aspect ratio and filling value of the original image
    vertical_padding = 0
    horizontal_padding = 0

    if src_aspect_ratio > 1.0:                  # 如果宽高比大于1.0，则宽度为正方形的边长，并计算水平填充值 If the aspect ratio is greater than 1.0, the width is the side length of the square and the horizontal filling value is calculated
        square_size = src_height
        horizontal_padding = int((src_width-square_size)/2)
    else:                                       # 否则，高度为正方形的边长，并计算垂直填充值 Otherwise, the height is the side length of the square and the vertical filling value is calculated
        square_size = src_width
        vertical_padding = int((src_height-square_size)/2)

    # 对图像进行裁剪，使其变为正方形  Crop the image to make it square
    cropped = frame[vertical_padding:vertical_padding+square_size,
                    horizontal_padding:horizontal_padding+square_size]
    return cropped
