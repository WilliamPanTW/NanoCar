import torch
import torchvision
import numpy as np
import time
import cv2
import os
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
from torch2trt import torch2trt
from torch2trt import TRTModule

from autopilot_utils import preprocess_image, center_crop_square
from autopilot_model import AutopilotModel

# TODO: set your paths # TODO: 设置你的路径
MODELS_DIR = ""
NAME = ""
MODEL_PATH = MODELS_DIR + NAME + ".pth"
MODEL_PATH_TRT = MODELS_DIR + NAME + "_trt.pth"

STEERING_OFFSET = 0.035
THROTTLE_GAIN = 0.8

CAMERA_WIDTH = 448
CAMERA_HEIGHT = 336

FRAME_SIZE = 224
FRAME_CHANNELS = 3

SHOW_LOGS = False

# Model # 模型
if os.path.isfile(MODEL_PATH_TRT): 	
	# 如果已经存在TensorRT模型，则加载它  Load pre-trained TRT model if available
	model_trt = TRTModule() 								# 创建一个空的 TensorRT 模型对象  Create an empty TensorRT model object
	model_trt.load_state_dict(torch.load(MODEL_PATH_TRT)) 	# 加载预训练好的 TensorRT 模型的参数和权重 Load the parameters and weights of the pre-trained TensorRT model
else: 
	# 否则，加载PyTorch模型并将其转换为TensorRT模型  Otherwise, load the PyTorch model and convert it to the TensorRT model
	model = AutopilotModel(pretrained=False) 			# 创建一个新的 AutopilotModel 对象，并指定不加载预训练的权重
	model.load_from_path(MODEL_PATH) 					# 从指定路径加载模型的参数和权重
	model.eval()										# 将模型设置为评估模式，即禁用 Dropout 和批量归一化层的更新

	x = torch.ones((1, FRAME_CHANNELS, FRAME_SIZE, FRAME_SIZE)).cuda() 	#创建一个输入张量 x, 并将其移动到 CUDA 设备上
	model_trt = torch2trt(model, [x], fp16_mode=True)					#将 PyTorch 模型转换为 TensorRT 模型
	torch.save(model_trt.state_dict(), MODEL_PATH_TRT)					#将转换后的 TensorRT 模型的参数和权重保存到指定的路径 MODEL_PATH_TRT

try:
	# Car 小车
	car = NvidiaRacecar()
	car.throttle_gain = THROTTLE_GAIN
	car.steering_offset = STEERING_OFFSET

	# Camera 摄像头
	camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

	# Control Loop 控制循环
	while True:
		if SHOW_LOGS:
			start_time = time.time()
		
		camera_frame = camera.read()																# 读取摄像头帧  read camera frame
		cropped_frame = center_crop_square(camera_frame)											# 裁剪帧为正方形  Crop frame to a square
		resized_frame = cv2.resize(cropped_frame, (FRAME_SIZE, FRAME_SIZE))							# 调整帧大小  Resize cropped frame to desired size
		preprocessed_frame = preprocess_image(resized_frame)										# 图像预处理  Preprocess the resized frame
		output = model_trt(preprocessed_frame).detach().clamp(-1.0, 1.0).cpu().numpy().flatten()	# 使用TensorRT模型进行推理  Pass the preprocessed frame through the model

		steering = float(output[0])																	# 获取转向值 Extract steering and throttle values from the output
		car.steering = STEERING_OFFSET																

		throttle = float(output[1])																	# 获取油门值 Get throttle value
		car.throttle = throttle

		# 打印帧率、转向值和油门值 Calculate and print frames per second (fps), steering, and throttle values
		if SHOW_LOGS:
			fps = int(1/(time.time()-start_time))				
			print("fps: " + str(int(fps)) + ", steering: " + str(steering) + ", throttle: " + str(throttle), end="\r")
   
# 在捕捉到键盘中断时停止小车  Stop the car and exit the program gracefully if interrupted by keyboard
except KeyboardInterrupt: 
	car.throttle = 0.0
	car.steering = 0.0
	raise SystemExit

	
