import torch
import torchvision
import os
import glob
import PIL.Image
from PIL import ImageFilter
import torch.utils.data
import cv2
import numpy as np

from autopilot_utils import center_crop_square

class AutopilotDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory,
                 frame_size, 
                 transform=None,
                 random_noise=False,
                 random_blur=False,
                 random_horizontal_flip=False,
                 random_color_jitter=False,
                 keep_images_in_ram=False): # requires lots of RAM but significantly speeds up IO, otherwise stores image paths(要大量内存但显著加快IO速度，否则只存储图像路径）

        super(AutopilotDataset, self).__init__()
        
        # 初始化数据集属性
        self.frame_size = frame_size
        self.transform = transform
        self.random_noise = random_noise
        self.random_blur = random_blur
        self.random_horizontal_flip = random_horizontal_flip
        self.random_color_jitter = random_color_jitter
        self.keep_images_in_ram = keep_images_in_ram
        
        # 初始化数据列表
        self.data = []

        # 打开注释文件并逐行读取
        with open(directory + "annotations.csv", 'r') as annotations:
            for line in annotations:
                # 从行中提取名称、转向和油门值
                name, steering, throttle = line.split(",") 
                image = directory+name+'.jpg'

                # 检查图像文件是否存在且不为空
                if os.path.isfile(image) and os.stat(image).st_size > 0:
                    # 如果keep_images_in_ram为True，则加载并准备图像；否则仅存储图像路径
                    if self.keep_images_in_ram:
                        image = self.load_and_prepare_image_from_path(image)
                    self.data.append((name, image, steering, throttle))
                    
            annotations.close()
        print("Generated dataset of " + str(len(self.data)) + " items") # print出项目的数据集
    
    # 定义并且返回数据集的长度      
    def __len__(self):
        return len(self.data)
    
    # 将element 跟 idx 赋值给变量 item
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 解包4项目元组
        name, image, steering, throttle = item

        # 如果keep_images_in_ram为False，则加载并准备图像
        if not self.keep_images_in_ram:
            image = self.load_and_prepare_image_from_path(image)
        
        # 将转向和油门值转换为浮点数
        steering = float(steering)
        throttle = float(throttle)
        
        # 如果random_blur为True且通过概率检查，则应用随机模糊
        if self.random_blur and float(np.random.random(1)) > 0.5:
            image = image.filter(ImageFilter.BLUR)
        
        #如果 random_noise 为 True，并且随机生成的一个浮点数大于 0.5的应用随机噪声    
        if self.random_noise and float(np.random.random(1)) > 0.5:
            output = np.copy(np.array(image)) #创建 output 变量，通过将图像转换为 NumPy 数组来获取图像数据的副本
    
            amount = 0.1 #创建output变量，通过将图像转换为NumPy数组来获取图像数据的副本
        
            nb_salt = np.ceil(amount * output.size * 0.5) #计算 nb_salt，表示添加盐噪声的像素数量,并向上取整

            #生成 coords 变量，包含了 nb_salt 个随机坐标，用于确定将要添加盐噪声的像素位置
            coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape] 
            output[tuple(coords)] = 1.0

            #计算 nb_pepper 表示添加椒噪声的像素数量
            nb_pepper = np.ceil(amount* output.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
            output[tuple(coords)] = 0.0
            
            #将修改后的 NumPy 数组转换回 PIL 图像对象，并将其赋值给变量 image
            image = PIL.Image.fromarray(output)
        
        #如果 random_horizontal_flip 为 True，并且随机生成的一个浮点数大于 0.5
        if self.random_horizontal_flip and float(np.random.random(1)) > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT) #将图像水平翻转
            steering = -steering    #获得反转转向值
            
        transforms = [] #根据配置的参数创建变换列表 transforms

        #如果 random_color_jitter 为 True，则添加颜色抖动变换
        if self.random_color_jitter:
            transforms = [torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25, saturation=0.25)]
            
        transforms += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        
        #通过调用torchvision.transforms.Compose(), 将所有的变换组合起来,并将其应用于图像，将结果保存回 image 变量                            
        composed_transforms = torchvision.transforms.Compose(transforms)
        image = composed_transforms(image)
        
        #返回样本的名称 name、处理后的图像数据 image，以及转向值 steering 和油门值 throttle 的张量表示
        return name, image, torch.Tensor([steering, throttle])

    #加载图像，并对其进行预处理，包括颜色空间转换、裁剪和调整大小后并返回image
    def load_and_prepare_image_from_path(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop_square(image)
        image = cv2.resize(image, (self.frame_size, self.frame_size))
        image = PIL.Image.fromarray(image)
        return image