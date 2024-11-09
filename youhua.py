import os
import cv2
import time
import numpy as np
from openvino.runtime import Core

# 1. 加载量化后的 OpenVINO 模型
core = Core()
model_path = "/root/yolov10-main/openvino_model/best_model.xml"  # 量化后的模型路径
compiled_model = core.compile_model(model_path, device_name="CPU")

# 2. 准备输入数据 - 从指定文件夹读取图片
image_folder = "/root/yolov10-main/data/test"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]  # 支持 jpg 和 png 格式

# 3. 初始化计时器和变量
inference_times = []  # 存储每张图片的推理时间

# 4. 循环遍历每张图片并进行推理
for image_file in image_files[:100]:  # 仅处理前 100 张图片
    # 读取图片并调整为模型所需的输入大小（假设模型输入是 640x640）
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))  # 根据模型输入要求调整图片大小
    image_input = np.transpose(image_resized, (2, 0, 1))  # 转换为 (C, H, W) 格式
    image_input = np.expand_dims(image_input, axis=0)  # 添加批量维度 (B, C, H, W)
    image_input = image_input.astype(np.float32)  # 转换为浮点数格式

    # 5. 记录开始时间
    start_time = time.time()

    # 6. 执行推理
    output = compiled_model([image_input])

    # 7. 记录结束时间
    end_time = time.time()

    # 8. 计算推理时间并添加到列表
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    # 输出每张图片的推理时间
    print(f"Image {image_file} inference time: {inference_time:.4f} seconds")

# 9. 计算平均推理时间
average_inference_time = np.mean(inference_times)
print(f"Average inference time for 100 images: {average_inference_time:.4f} seconds")

# 10. 计算每秒处理的图片数 (FPS)
fps = 1 / average_inference_time
print(f"Frames per second (FPS): {fps:.2f}")
