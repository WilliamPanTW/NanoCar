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
        
        # 初始化数据集属性 Initialize dataset parameters 
        self.frame_size = frame_size
        self.transform = transform
        self.random_noise = random_noise
        self.random_blur = random_blur
        self.random_horizontal_flip = random_horizontal_flip
        self.random_color_jitter = random_color_jitter
        self.keep_images_in_ram = keep_images_in_ram
        
        # 初始化数据列表 Initialize data list to store image information
        self.data = []

        # 打开注释文件并逐行读取 Open the annotations file
        with open(directory + "annotations.csv", 'r') as annotations:
            for line in annotations:
                # 从行中提取名称、转向和油门值  Split the line into name, steering, and throttle
                name, steering, throttle = line.split(",") 
                image = directory+name+'.jpg'

                # 检查图像文件是否存在且不为空 Check if the image file exists and is not empty
                if os.path.isfile(image) and os.stat(image).st_size > 0:
                    # 如果keep_images_in_ram为True，则加载并准备图像；否则仅存储图像路径 
                    # If keep_images_in_ram is True, load and prepare the image from path
                    if self.keep_images_in_ram:
                        image = self.load_and_prepare_image_from_path(image)
                    self.data.append((name, image, steering, throttle))
                    
            annotations.close()
        print("Generated dataset of " + str(len(self.data)) + " items") # print出项目的数据集 Print the number of items in the generated dataset
    
    # 定义并且返回数据集的长度  Defines and returns the length of the data set     
    def __len__(self):
        return len(self.data)
    
    # 将element 跟 idx 赋值给变量 item
    def __getitem__(self, idx):
        # Get an item from the data list
        item = self.data[idx]
        
        # 解包4项目元组  Unpack the item into name, image, steering, and throttle
        name, image, steering, throttle = item

        # 如果keep_images_in_ram为False，则加载并准备图像  Load and prepare the image from path
        if not self.keep_images_in_ram:
            image = self.load_and_prepare_image_from_path(image)
        
        # 将转向和油门值转换为浮点数 Converts steering and throttle values to floating-point numbers
        steering = float(steering)
        throttle = float(throttle)
        
        # 如果random_blur为True且通过概率检查，则应用随机模糊 If random_blur is True and passes the probability check, then random blur is applied
        if self.random_blur and float(np.random.random(1)) > 0.5:
            image = image.filter(ImageFilter.BLUR)
        
        # 如果 random_noise 为 True，并且随机生成的一个浮点数大于 0.5的应用随机噪声 
        # If random_noise is True and a floating point number greater than 0.5 is randomly generated, apply random noise   
        if self.random_noise and float(np.random.random(1)) > 0.5:
            output = np.copy(np.array(image)) # 创建 output 变量，通过将图像转换为 NumPy 数组来获取图像数据的副本 Creates the output variable to get a copy of the image data by converting the image to a NumPy array
    
            amount = 0.1 # 创建output变量，通过将图像转换为NumPy数组来获取图像数据的副本 Creates the output variable to get a copy of the image data by converting the image to a NumPy array
        
            nb_salt = np.ceil(amount * output.size * 0.5) # 计算 nb_salt，表示添加盐噪声的像素数量,并向上取整 Calculate nb_salt, representing the number of pixels to which salt noise is added, and round it up

            # 生成 coords 变量，包含了 nb_salt 个随机坐标，用于确定将要添加盐噪声的像素位置 
            # Generates a coords variable containing a random coordinate of nb_salt to determine the location of the pixel to which salt noise will be added
            coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape] 
            output[tuple(coords)] = 1.0

            # 计算 nb_pepper 表示添加椒噪声的像素数量  Calculate nb_pepper to represent the number of pixels to which pepper noise is added
            nb_pepper = np.ceil(amount* output.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
            output[tuple(coords)] = 0.0
            
            # 将修改后的 NumPy 数组转换回 PIL 图像对象，并将其赋值给变量 image  # Convert the modified image back to PIL Image
            image = PIL.Image.fromarray(output)
        
        #如果 random_horizontal_flip 为 True，并且随机生成的一个浮点数大于 0.5
        if self.random_horizontal_flip and float(np.random.random(1)) > 0.5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT) #将图像水平翻转 Flip the image horizontally
            steering = -steering    #获得反转转向值  Adjust the steering value accordingly
            
        transforms = [] # 根据配置的参数创建变换列表 transforms # transforms list Transforms are created based on configured parameters

        # 如果 random_color_jitter 为 True，则添加颜色抖动变换 apply color jitter transformation if random_color_jitter flag is True
        if self.random_color_jitter:
            transforms = [torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.25, saturation=0.25)]
            
        transforms += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        
        # 通过调用torchvision.transforms.Compose(), 将所有的变换组合起来,并将其应用于图像，将结果保存回 image 变量
        # Apply the composed transformations to the image                          
        composed_transforms = torchvision.transforms.Compose(transforms)
        image = composed_transforms(image)
        
        # 返回样本的名称 name、处理后的图像数据 image，以及转向值 steering 和油门值 throttle 的张量表示
        # Returns the name of the sample, the processed image data image, and a tensor representation of the steering value and throttle value
        return name, image, torch.Tensor([steering, throttle])

    # 加载图像，并对其进行预处理，包括颜色空间转换、裁剪和调整大小后并返回image
    #  Load and prepare the image from the given file path
    def load_and_prepare_image_from_path(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop_square(image)
        image = cv2.resize(image, (self.frame_size, self.frame_size))
        image = PIL.Image.fromarray(image)
        return image