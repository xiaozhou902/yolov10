from openvino.runtime import Core
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# 创建 Core 对象
core = Core()

# 读取模型
print("Loading model...")
model = core.read_model(model='/root/yolov10-main/openvino_model/best_model.xml')

# 编译模型
print("Compiling model...")
compiled_model = core.compile_model(model=model, device_name="CPU")  # 如果你有 GPU 设备，可以替换为 "GPU"

# 获取输入输出层
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

# 加载测试数据集
image_folder = '/root/yolov10-main/data/test'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 输出加载的图像文件列表
print(f"Found {len(image_files)} image files in the dataset.")

# 获取 ground truth 标签
def get_ground_truth(image_file):
    # 假设标签存储在 `/root/yolov10-main/data/output` 路径下
    label_file = image_file.replace('.jpg', '.xml')  # 将图像文件名后缀改为 .xml
    label_path = os.path.join('/root/yolov10-main/data/output', label_file)
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} does not exist!")
        return []  # 如果没有标签文件，则返回空列表

    # 解析 XML 文件
    tree = ET.parse(label_path)
    root = tree.getroot()

    ground_truth = []
    for obj in root.findall('object'):
        class_id = obj.find('name').text  # 标签类别
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        ground_truth.append({
            'class_id': class_id,
            'bbox': [xmin, ymin, xmax, ymax]  # 边界框坐标
        })
    
    return ground_truth

# 计算 mAP 或其他精度指标
def compute_mAP(predictions, ground_truths):
    print("Computing mAP...")
    # 这里可以加入 mAP 计算逻辑
    pass

predictions = []
ground_truths = []  # 真实标签

# 遍历图像文件进行推理
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"Processing image: {image_file}")
    
    # 读取图像并处理
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))  # 假设输入尺寸为640x640
    image_normalized = image_resized / 255.0
    image_input = np.transpose(image_normalized, (2, 0, 1))  # 转换为 CxHxW
    image_input = np.expand_dims(image_input, axis=0)  # 添加批次维度
    
    # 推理
    try:
        print("Running inference...")
        result = compiled_model([image_input])  # 调用 compiled_model 推理
    except Exception as e:
        print(f"Error during inference: {e}")
        continue
    
    # 获取预测结果
    prediction = result[output_layer].squeeze(0)  # 获取预测结果并去掉批次维度
    print(f"Predicted bounding boxes for {image_file}: {prediction}")

    # 假设输出是类似于 YOLO 的输出（bounding boxes, class ids, scores）
    prediction_boxes = prediction[prediction[:, 4] > 0.5]  # 只保留置信度大于0.5的框
    predictions.append(prediction_boxes)

    # 获取对应的真实标签
    ground_truth = get_ground_truth(image_file)
    print(f"Ground truth for {image_file}: {ground_truth}")
    ground_truths.append(ground_truth)

# 计算准确度、mAP 等
compute_mAP(predictions, ground_truths)
